SOURCES = $(wildcard *.cpp)
OBJDIR   = ./obj
OBJECTS  = $(addprefix $(OBJDIR)/, $(SOURCES:.cpp=.o))
TARGETS = $(SOURCES:.cpp=.exe)
DEPENDS = $(OBJECTS:.o=.d)

CXX      = g++
CXXFLAGS = -g -MMD -MP -Wall -Wextra -Winit-self -Wno-unused-parameter -std=c++20 -O3
RM       = rm -f
LDFLAGS  =
INCLUDE  = 
# INCLUDE  = -I ac-library

.PHONY:run
run:$(TARGETS) $(OBJECTS)

%.exe: $(OBJDIR)/%.o
	$(CXX) -o $@ $< $(LDFLAGS)
	# ./$@

$(OBJDIR)/%.o: %.cpp
	@if [ ! -d $(OBJDIR) ];\
	then echo "mkdir -p $(OBJDIR)";mkdir -p $(OBJDIR);\
	fi
	$(CXX) $(CXXFLAGS) $(INCLUDE) -o $@ -c $<

#クリーンして実行
.PHONY: all
all: clean run

#クリーン
.PHONY:clean
clean: 
	$(RM) $(TARGETS) $(OBJECTS) $(DEPENDS)


-include $(DEPENDS)
