#include "MemoryMonitor.h"


// Gets snapshot of the memory used and check if it is higher than the recorded peak
void MemoryMonitor::updatePeakMemory()
{
	std::chrono::milliseconds delay(200);
	size_t currentMem;

	while (enabled) {
		currentMem = MemoryMonitor::getCurrentMemory();
		if (currentMem > peakMemory) {
			peakMemory = currentMem;
		}
		std::this_thread::sleep_for(delay);
	}
	return;
}

MemoryMonitor::MemoryMonitor()
{
	peakMemory = 0;
	enabled = true;
}

size_t MemoryMonitor::getPeakMemory()
{
	if (peakMemory == 0) {
		return MemoryMonitor::getCurrentMemory();
	}
	return peakMemory;
}


size_t MemoryMonitor::getCurrentMemory()
{
#if defined(_WIN32)
	/* Windows -------------------------------------------------- */
	PROCESS_MEMORY_COUNTERS_EX info;
	GetProcessMemoryInfo(GetCurrentProcess(), reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&info), sizeof(info));
	return (size_t)info.PrivateUsage;

#elif defined(__APPLE__) && defined(__MACH__)
	/* OSX ------------------------------------------------------ */
	struct mach_task_basic_info info;
	mach_msg_type_number_t infoCount = MACH_TASK_BASIC_INFO_COUNT;
	if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
		(task_info_t)&info, &infoCount) != KERN_SUCCESS)
		return (size_t)0L;      /* Can't access? */
	return (size_t)info.resident_size;

#elif defined(__linux__) || defined(__linux) || defined(linux) || defined(__gnu_linux__)
	/* Linux ---------------------------------------------------- */
	long rss = 0L;
	FILE* fp = NULL;
	if ((fp = fopen("/proc/self/statm", "r")) == NULL)
		return (size_t)0L;      /* Can't open? */
	if (fscanf(fp, "%*s%ld", &rss) != 1)
	{
		fclose(fp);
		return (size_t)0L;      /* Can't read? */
	}
	fclose(fp);
	return (size_t)rss * (size_t)sysconf(_SC_PAGESIZE);

#else
	/* AIX, BSD, Solaris, and Unknown OS ------------------------ */
	return (size_t)0L;          /* Unsupported. */
#endif
}

void MemoryMonitor::startMonitoring()
{
	peakMemory = 0;
}

void MemoryMonitor::stopMonitoring()
{
	enabled = false;
}

MemoryMonitor::~MemoryMonitor()
{
}
