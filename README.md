
### Installation

- You first need to build the direct fortran library. For this `cd` into
  `utils/direct_fortran` and run `bash make_direct.sh`. You will need a fortran compiler
  such as gnu95. Once this is done, you can run `simple_direct_test.py` to make sure that
  it was installed correctly.
- Run `source set_up_gittins` to set up all environment variables.
- To test the installation, run `bash run_all_tests.sh`. Some of the tests are
  probabilistic and could fail at times. If this happens, run the same test several times
  and make sure it isn't consistently failing.
