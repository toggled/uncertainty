#pragma once
#include <chrono>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <thread>
#if defined(_WIN32)
#define NOMINMAX
#include <windows.h>
#include <psapi.h>

#elif defined(__unix__) || defined(__unix) || defined(unix) || (defined(__APPLE__) && defined(__MACH__))
#include <unistd.h>
#include <sys/resource.h>


#if defined(__APPLE__) && defined(__MACH__)
#include <mach/mach.h>

#elif (defined(_AIX) || defined(__TOS__AIX__)) || (defined(__sun__) || defined(__sun) || defined(sun) && (defined(__SVR4) || defined(__svr4__)))
#include <fcntl.h>
#include <procfs.h>

#elif defined(__linux__) || defined(__linux) || defined(linux) || defined(__gnu_linux__)
#include <stdio.h>
#endif

#else
#error "Cannot define getPeakRSS( ) or getCurrentRSS( ) for an unknown OS."
#endif


class MemoryMonitor
{
private:
	size_t peakMemory;
	bool enabled;
public:
	MemoryMonitor();
	size_t getPeakMemory();
	static size_t getCurrentMemory();
	void startMonitoring();
	void stopMonitoring();
	void updatePeakMemory();
	~MemoryMonitor();
};

