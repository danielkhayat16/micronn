CXX = g++
CXXFLAGS = -Wall -O2 -std=c++11

TARGET = micronn
SRC = main.cpp layers.cpp mnist_loader.cpp train.cpp model.cpp inference.cpp
OBJ = $(SRC:.cpp=.o)

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $<

clean:
	rm -f *.o $(TARGET)
