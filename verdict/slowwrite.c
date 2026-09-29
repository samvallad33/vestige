#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

static int fd_is_seg(int fd) {
    char linkpath[64];
    char target[512];
    snprintf(linkpath, sizeof linkpath, "/proc/self/fd/%d", fd);
    ssize_t n = readlink(linkpath, target, sizeof target - 1);
    if (n < 0) {
        return 0;
    }
    target[n] = 0;
    return strstr(target, ".seg") != NULL;
}

ssize_t write(int fd, const void *buf, size_t count) {
    static ssize_t (*real_write)(int, const void *, size_t) = NULL;
    if (!real_write) {
        real_write = dlsym(RTLD_NEXT, "write");
    }
    if (fd_is_seg(fd)) {
        usleep(80000);
    }
    return real_write(fd, buf, count);
}
