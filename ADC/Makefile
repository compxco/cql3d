# Makefile that builts a library lib$(LIB).a from all
# of the Fortran files found in the current directory.
# Usage: make LIB=<libname>
# Pearu
LIB=ADC
OBJ=$(patsubst %.f,%.o,$(shell ls ADC/*.f))
all: lib$(LIB).a
$(OBJ):
	$(FC) -c $(FFLAGS) $(patsubst ADC/%.o,ADC/%.f,ADC/$(@F)) -o $@
lib$(LIB).a: $(OBJ)
	$(AR) rus ADC/lib$(LIB).a $?
clean:
	rm -f ADC/*.o
	rm -f ADC/*.a