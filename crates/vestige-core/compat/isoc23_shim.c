/*
 * C23 strto* shims for the prebuilt ONNX Runtime static archive.
 *
 * ort-sys download-binaries (pyke CDN, currently ms@1.23.2) ships
 * libonnxruntime.a compiled on Ubuntu 24.04. Those objects call
 * __isoc23_strtol / __isoc23_strtoll / __isoc23_strtoull (and, on
 * newer archives, the _l variants and strtoul), which glibc only
 * exports from 2.38.
 *
 * Ubuntu 22.04 is glibc 2.35 and Debian 12 is glibc 2.36, so a
 * linux-gnu vestige-mcp linked against that archive dies at process
 * start with GLIBC_2.38 not found — the published 2.3.0 Linux asset
 * and the #174 / #175 MCP spec-test crashes.
 *
 * Forward to the classic strto* functions. The only C23 difference is
 * accepting a 0b/0B prefix when base is 0 or 2; ONNX Runtime parses
 * decimal/hex config, so this is safe.
 *
 * Do not include <stdlib.h>. glibc >= 2.38 redirects strtol to
 * __isoc23_strtol, which would recurse. Bind the pre-C23 versions
 * with .symver so the shim stays correct even if this file is
 * compiled on a 2.38+ host.
 */

extern long vestige_strtol(const char *nptr, char **endptr, int base);
extern long long vestige_strtoll(const char *nptr, char **endptr, int base);
extern unsigned long vestige_strtoul(const char *nptr, char **endptr, int base);
extern unsigned long long vestige_strtoull(const char *nptr, char **endptr, int base);
extern long long vestige_strtoll_l(const char *nptr, char **endptr, int base, void *loc);
extern unsigned long long vestige_strtoull_l(const char *nptr, char **endptr, int base, void *loc);

__asm__(".symver vestige_strtol,strtol@GLIBC_2.2.5");
__asm__(".symver vestige_strtoll,strtoll@GLIBC_2.2.5");
__asm__(".symver vestige_strtoul,strtoul@GLIBC_2.2.5");
__asm__(".symver vestige_strtoull,strtoull@GLIBC_2.2.5");
__asm__(".symver vestige_strtoll_l,strtoll_l@GLIBC_2.3.3");
__asm__(".symver vestige_strtoull_l,strtoull_l@GLIBC_2.3.3");

long __isoc23_strtol(const char *nptr, char **endptr, int base)
{
	return vestige_strtol(nptr, endptr, base);
}

long long __isoc23_strtoll(const char *nptr, char **endptr, int base)
{
	return vestige_strtoll(nptr, endptr, base);
}

unsigned long __isoc23_strtoul(const char *nptr, char **endptr, int base)
{
	return vestige_strtoul(nptr, endptr, base);
}

unsigned long long __isoc23_strtoull(const char *nptr, char **endptr, int base)
{
	return vestige_strtoull(nptr, endptr, base);
}

long long __isoc23_strtoll_l(const char *nptr, char **endptr, int base, void *loc)
{
	return vestige_strtoll_l(nptr, endptr, base, loc);
}

unsigned long long __isoc23_strtoull_l(const char *nptr, char **endptr, int base, void *loc)
{
	return vestige_strtoull_l(nptr, endptr, base, loc);
}
