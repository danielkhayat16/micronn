CXX = g++
CXXFLAGS = -Wall -O2 -std=c++11
OBJS = layers.o mnist_loader.o inference.o model.o

all: micronn draw

micronn: main.o $(OBJS) train.o
	$(CXX) $(CXXFLAGS) -o $@ $^

draw: draw.o $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ -lSDL2

main.o: main.cpp
	$(CXX) $(CXXFLAGS) -c $<

draw.o: draw.cpp
	$(CXX) $(CXXFLAGS) -c $<

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $<

clean:
	rm -f *.o micronn draw
