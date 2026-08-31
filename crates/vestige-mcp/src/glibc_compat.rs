//! glibc 2.38 C23 `strtol` compatibility shims (Linux/gnu only).
//!
//! # The bug this fixes
//!
//! Released `x86_64-unknown-linux-gnu` binaries up to and including v2.3.0
//! aborted at startup on any distro older than Ubuntu 24.04:
//!
//! ```text
//! ./vestige-mcp: /lib/x86_64-linux-gnu/libc.so.6: version `GLIBC_2.38' not found
//! ```
//!
//! The server died before it could answer a single MCP request, which is what
//! issues #174 and #175 actually reported: every "spec conformance violation"
//! in those two reports was the harness observing a process that had already
//! exited, not a protocol defect.
//!
//! # Why it happened
//!
//! glibc 2.38 added the C23 flavours of the string-to-integer functions, which
//! additionally accept `0b`/`0B` binary literals. When a translation unit is
//! compiled in C23 mode against glibc >= 2.38 headers, `<stdlib.h>` quietly
//! redirects the call with an asm label:
//!
//! ```c
//! extern long int strtol (...) __asm__ ("__isoc23_strtol");
//! ```
//!
//! so the object file ends up importing `__isoc23_strtol@GLIBC_2.38` instead of
//! plain `strtol@GLIBC_2.2.5`.
//!
//! We do not compile anything in C23 mode ourselves. The references come from
//! the prebuilt `libonnxruntime.a` that `ort-sys` downloads and that the default
//! `ort-download` feature links statically into the binary: it carries 30
//! undefined `__isoc23_*` references in objects that are always pulled in
//! (`allocation_planner`, `inference_session`, `cast_op`, `parser`, `re2`, ...).
//!
//! That detail rules out the obvious fixes. Building on an older runner does not
//! help, because those references live in a prebuilt archive we do not compile,
//! and linking it against a pre-2.38 glibc fails outright with "undefined
//! reference to `__isoc23_strtol`". Pinning `ubuntu-22.04` would also be a dead
//! end on its own: that image begins deprecation on 2026-09-17.
//!
//! # The fix
//!
//! Define the symbols ourselves and forward to the classic entry points. Rust
//! `extern "C"` declarations are not affected by the `<stdlib.h>` redirect, so
//! `strtol` here binds to plain `strtol@GLIBC_2.2.5`. The linker resolves
//! ONNX Runtime's undefined references against these definitions, no
//! `GLIBC_2.38` import is emitted, and the floor drops to `GLIBC_2.34` (set by
//! the `pthread_*`/`dlopen` symbols that moved into libc in glibc 2.34). That
//! covers RHEL/Rocky/Alma 9, Amazon Linux 2023, Ubuntu 22.04+ and Debian 12+.
//!
//! The only behavioural difference is that the C23 `0b1010` binary-literal form
//! is not recognised, which matches what these call sites did on every glibc
//! before 2.38 anyway.
//!
//! `scripts/check-glibc-floor.sh` enforces the invariant in CI so a future
//! dependency bump cannot silently reintroduce the crash.

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

unsafe extern "C" {
    fn __errno_location() -> *mut c_int;
}

/// `ENOSYS`, the asm-generic value shared by x86_64 and aarch64 — the only
/// Linux targets Vestige releases. It differs on alpha/mips/parisc/sparc, so a
/// new Linux target on one of those would need a `cfg` here.
const ENOSYS: c_int = 38;

/// glibc 2.39 `pidfd_getpid`, defined locally so no `GLIBC_2.39` import is
/// emitted.
///
/// Rust std reaches these through its `weak!` macro, which on ELF is
/// `#[linkage = "extern_weak"]`. The symbol really is weak — but the linker
/// still records a *non-weak* `GLIBC_2.39` version need (`readelf -V` shows
/// `Flags: none`), and the loader refuses to start the process on any older
/// glibc regardless of the symbol's binding. That is why the shipped v2.3.0
/// `vestige` CLI, which is the only binary using `std::process::Command`,
/// already failed on anything below Ubuntu 24.04 even where `vestige-mcp` ran.
///
/// Nothing here ever calls these: std only reaches for pidfd when
/// `Command::create_pidfd(true)` was set (the unstable `linux_pidfd` API,
/// default false). The definitions exist to satisfy the linker. If Vestige ever
/// does opt into `create_pidfd`, the `ENOSYS` answers below make std fall back
/// to fork/exec — correct, but no longer race-free against pid recycling.
///
/// # Safety
///
/// Same contract as `pidfd_getpid(2)`: returns -1 and sets `errno`.
#[unsafe(no_mangle)]
pub extern "C" fn pidfd_getpid(_pidfd: c_int) -> c_int {
    unsafe { *__errno_location() = ENOSYS };
    -1
}

/// glibc 2.39 `pidfd_spawnp`, defined locally for the same reason.
///
/// The `posix_spawn` family returns the error number directly rather than
/// setting `errno`, and std checks specifically for `ENOSYS` to select the
/// fork/exec path.
#[unsafe(no_mangle)]
pub extern "C" fn pidfd_spawnp(
    _pidfd: *mut c_int,
    _file: *const c_char,
    _file_actions: *const c_void,
    _attrp: *const c_void,
    _argv: *const *mut c_char,
    _envp: *const *mut c_char,
) -> c_int {
    ENOSYS
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
    fn pidfd_stubs_report_enosys_so_std_falls_back_to_fork_exec() {
        assert_eq!(
            pidfd_spawnp(
                std::ptr::null_mut(),
                c"/bin/true".as_ptr(),
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
            ),
            ENOSYS,
            "posix_spawn-family calls return the error number directly"
        );

        assert_eq!(pidfd_getpid(-1), -1);
        unsafe {
            assert_eq!(
                *__errno_location(),
                ENOSYS,
                "pidfd_getpid reports via errno"
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
}
