//! Symbols the statically linked ONNX Runtime archive imports from a newer
//! glibc and libstdc++ than the release floor provides (Linux/gnu only).
//!
//! # What breaks without this
//!
//! `ort-sys` downloads a prebuilt `libonnxruntime.a` and links it into every
//! Vestige binary. The archive was compiled with a recent toolchain, and its
//! objects import two families of symbols that the release container (Ubuntu
//! 22.04: glibc 2.35, libstdc++ from GCC 12) does not have:
//!
//! - `__isoc23_strtol`, `__isoc23_strtoll`, `__isoc23_strtoull`: glibc 2.38
//!   added C23 flavours of the string-to-integer functions (they additionally
//!   accept `0b` binary literals). A translation unit compiled in C23 mode
//!   against glibc >= 2.38 headers gets its plain `strtol` call silently
//!   redirected by an asm label to `__isoc23_strtol@GLIBC_2.38`.
//! - `__cxa_call_terminate`: GCC 13 emits calls to this libsupc++ helper when
//!   an exception escapes a `noexcept` region; libstdc++ exports it only from
//!   `CXXABI_1.3.15`, which Ubuntu 22.04, Debian 12 and RHEL 9 all lack.
//!
//! The x86_64 archive happens not to import either family, which is why the
//! x86_64 release job never saw this. The aarch64 archive imports all four
//! (123 `__isoc23_*` and 180 `__cxa_call_terminate` references in ORT 1.23.2),
//! so the first aarch64 release build failed at link time with "undefined
//! reference". Nothing we compile is involved; the references live in a
//! prebuilt archive, so neither a compiler flag nor an older runner can remove
//! them, and a newer runner would only move the problem to users' machines as
//! `version GLIBC_2.38 not found` at startup.
//!
//! # The fix
//!
//! Define the symbols here and forward to what every supported glibc and
//! libstdc++ already has. Rust `extern "C"` declarations are not subject to
//! the `<stdlib.h>` redirect, so `strtol` below binds to the classic entry
//! point. The linker resolves the archive's undefined references against these
//! definitions and emits no `GLIBC_2.38` or `CXXABI_1.3.15` version need. The
//! module is compiled into each binary root (`main.rs`, `bin/cli.rs`,
//! `bin/restore.rs`) rather than into the library, so the definitions are
//! always part of the final link instead of depending on archive member
//! selection.
//!
//! Behavioural differences, both deliberate:
//!
//! - The C23 `0b1010` binary-literal form is not recognised, which is what
//!   these call sites did on every glibc before 2.38 anyway.
//! - `__cxa_call_terminate` aborts the process directly instead of running a
//!   `std::set_terminate` handler. Neither Vestige nor ONNX Runtime installs
//!   one, and the default handler is `abort()`; this also keeps the shim free
//!   of any libstdc++ dependency, so builds without C++ code link unchanged.
//!
//! `scripts/check-linux-glibc.sh` keeps the ceiling honest in CI, and the
//! release job then starts each binary on Ubuntu 22.04 and Debian 12.

use std::os::raw::{c_char, c_int, c_long, c_longlong, c_ulong, c_ulonglong, c_void};

unsafe extern "C" {
    fn strtol(nptr: *const c_char, endptr: *mut *mut c_char, base: c_int) -> c_long;
    fn strtoll(nptr: *const c_char, endptr: *mut *mut c_char, base: c_int) -> c_longlong;
    fn strtoul(nptr: *const c_char, endptr: *mut *mut c_char, base: c_int) -> c_ulong;
    fn strtoull(nptr: *const c_char, endptr: *mut *mut c_char, base: c_int) -> c_ulonglong;
}

/// C23 `strtol`, forwarded to the pre-2.38 entry point.
///
/// # Safety
///
/// Same contract as `strtol(3)`: `nptr` must point at a NUL-terminated string
/// and `endptr` must be null or point at a writable `*mut c_char`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __isoc23_strtol(
    nptr: *const c_char,
    endptr: *mut *mut c_char,
    base: c_int,
) -> c_long {
    unsafe { strtol(nptr, endptr, base) }
}

/// C23 `strtoll`, forwarded to the pre-2.38 entry point.
///
/// # Safety
///
/// Same contract as `strtoll(3)`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __isoc23_strtoll(
    nptr: *const c_char,
    endptr: *mut *mut c_char,
    base: c_int,
) -> c_longlong {
    unsafe { strtoll(nptr, endptr, base) }
}

/// C23 `strtoul`, forwarded to the pre-2.38 entry point.
///
/// # Safety
///
/// Same contract as `strtoul(3)`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __isoc23_strtoul(
    nptr: *const c_char,
    endptr: *mut *mut c_char,
    base: c_int,
) -> c_ulong {
    unsafe { strtoul(nptr, endptr, base) }
}

/// C23 `strtoull`, forwarded to the pre-2.38 entry point.
///
/// # Safety
///
/// Same contract as `strtoull(3)`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __isoc23_strtoull(
    nptr: *const c_char,
    endptr: *mut *mut c_char,
    base: c_int,
) -> c_ulonglong {
    unsafe { strtoull(nptr, endptr, base) }
}

/// GCC 13's `__cxa_call_terminate(_Unwind_Exception*)`, exported by libstdc++
/// only from `CXXABI_1.3.15`.
///
/// The compiler emits a call to it when an exception propagates out of a
/// `noexcept` region; the only legal outcome is termination. libsupc++ marks
/// the exception caught and calls `std::terminate()`, whose default handler is
/// `abort()`. This version aborts directly (see the module docs for why), so
/// the exception header is not inspected.
#[unsafe(no_mangle)]
pub extern "C" fn __cxa_call_terminate(_exception: *mut c_void) -> ! {
    std::process::abort()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ptr;

    #[test]
    fn forwards_decimal_and_hex_like_strtol() {
        unsafe {
            assert_eq!(__isoc23_strtol(c"1234".as_ptr(), ptr::null_mut(), 10), 1234);
            assert_eq!(__isoc23_strtol(c"0x2a".as_ptr(), ptr::null_mut(), 0), 42);
            assert_eq!(
                __isoc23_strtoll(c"-9000".as_ptr(), ptr::null_mut(), 10),
                -9000
            );
            assert_eq!(
                __isoc23_strtoul(c"4294967295".as_ptr(), ptr::null_mut(), 10),
                4294967295
            );
            assert_eq!(
                __isoc23_strtoull(c"18446744073709551615".as_ptr(), ptr::null_mut(), 10),
                u64::MAX
            );
        }
    }

    #[test]
    fn reports_the_unparsed_tail_through_endptr() {
        unsafe {
            let input = c"77rest";
            let mut end: *mut c_char = ptr::null_mut();
            assert_eq!(__isoc23_strtol(input.as_ptr(), &mut end, 10), 77);
            assert_eq!(
                std::ffi::CStr::from_ptr(end).to_str().unwrap(),
                "rest",
                "endptr must point at the first unconsumed byte"
            );
        }
    }

    /// The shim must terminate the process, and with SIGABRT specifically, so
    /// a crash inside ONNX Runtime looks exactly like the libstdc++ original.
    /// Runs itself as a child: the child hits the probe env var and calls the
    /// shim; the parent checks how it died.
    #[test]
    fn calling_cxa_call_terminate_aborts_the_process() {
        use std::os::unix::process::ExitStatusExt;
        const PROBE: &str = "VESTIGE_GLIBC_COMPAT_ABORT_PROBE";
        const SIGABRT: i32 = 6;

        if std::env::var_os(PROBE).is_some() {
            __cxa_call_terminate(ptr::null_mut());
        }

        let exe = std::env::current_exe().expect("test binary path");
        let status = std::process::Command::new(exe)
            .arg("--exact")
            .arg("glibc_compat::tests::calling_cxa_call_terminate_aborts_the_process")
            .env(PROBE, "1")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .expect("spawn probe child");
        assert!(!status.success(), "the probe child must not exit cleanly");
        assert_eq!(
            status.signal(),
            Some(SIGABRT),
            "the probe child must die from SIGABRT, got {status:?}"
        );
    }
}
