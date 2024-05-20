#ifndef __MAIN_H__
#define __MAIN_H__

#include <iostream>


// ----------------------------------------------
template<class T>
void printArray(T* data, size_t length, bool doflush = true)
// ----------------------------------------------
{
	for(size_t i = 0; i < length; i++)
	{
		std::cout << data[i] << " ";
	}
	std::cout << "\n";
	if(doflush){ std::cout << std::flush; }
}


#endif // __MAIN_H__

