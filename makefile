# Compiler settings - Can be customized.
CXX = g++
CXXFLAGS = -Wall -std=c++17 -I./gui -I./gym $(shell sdl2-config --cflags)
LDFLAGS = $(shell sdl2-config --libs)

# Find all cpp files in the current directory and subdirectories
SOURCES = $(wildcard *.cpp) $(wildcard gui/*.cpp) $(wildcard gym/*.cpp)
# Define object files based on the source files
OBJECTS = $(SOURCES:.cpp=.o)
# Name of the executable
TARGET = main

# The first rule is the one executed when no parameters are fed into the Makefile
all: $(TARGET)

# Link the target with all objects files
$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

# Compile each source file to an object
# The -c flag says to generate the object file, 
# the -o $@ says to put the output of the compilation in the file named on the left side of the :, 
# the $< is the first item in the dependencies list, and the CXXFLAGS are the flags I chose to use in compiling.
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean up
clean:
	rm -f $(OBJECTS) $(TARGET)

# Rule for making a clean build
rebuild: clean all

.PHONY: all clean rebuild