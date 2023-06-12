from distutils.core import setup, Extension
import sysconfig

def main():
    CFLAGS = ['-g', '-Wall', '-Wno-unused-function', '-Wconversion', '-std=c99', '-fopenmp', '-mavx', '-mfma', '-pthread', '-O3']
    LDFLAGS = ['-fopenmp']
    setup(name="dumbpy",
          version="0.0.1",
          description="numc matrix operations with out optimization",
          ext_modules=[
            Extension("dumbpy",
                      sources=["src/numc.c", "src/matrix.c"],
                      extra_compile_args=CFLAGS,
                      extra_link_args=LDFLAGS,
                      language='c')
          ])


if __name__ == "__main__":
    main()
