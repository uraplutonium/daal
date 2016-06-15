# Frequently Asked Questions

This document provides answers to frequently asked questions about Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL).

If you have a question that is not covered here, let us know via [https://github.com/01org/daal/issues](https://github.com/01org/daal/issues) or [Intel(R) DAAL Forum](https://software.intel.com/en-us/forums/intel-data-analytics-acceleration-library).

## Building Intel(R) DAAL from the sources

### Windows*

* During the build process I can see errors reported by the `find` command. Any ideas on what can cause them?

	It is likely caused by executing the MSYS2* `find` command on an FAT32 file system. Building Intel DAAL on NTFS should not cause these issues.

* What can cause Makefile* to fail with a **fork: Resource temporarily unavailable** error?

	This error can be caused by executing the MSYS2* `make` command with a large number of threads. To solve this issue, try using `make` with fewer threads, for example: `make daal -j 4 PLAT=win32e`.
