Code for the paper: http://proceedings.mlr.press/v80/sen18a/sen18a.pdf
Please cite the above paper if using this code base for a publication. 
### Installation

- You first need to build the direct fortran library. For this `cd` into
  `utils/direct_fortran` and run `bash make_direct.sh`. You will need a fortran compiler
  such as gnu95. Once this is done, you can run `simple_direct_test.py` to make sure that
  it was installed correctly.
- Run `source set_up_gittins` to set up all environment variables.
- To test the installation, run `bash run_all_tests.sh`. Some of the tests are
  probabilistic and could fail at times. If this happens, run the same test several times
  and make sure it isn't consistently failing.
 - Python packages required are numpy, pandas, scikit-learn, matplotlib, multiprocessing, brewer2mpl


  __Running Synthetic Examples__

  1. In order to run the synthetic examples, go to the file `MFPDOO/experiments_synthetic.py` 
  2. Set `mfobject` as the desired function. For instance it has been set as `borehole`.
  3. Now from the root directory of the project run `python MFPDOO/experiments_synthetic.py`

  __Running SVM parameter tuning example__

  1. For hyper-paremeter tuning we need to convert the parameter tuning problem into a multi-fidelity black box optimization object `mfobject` like those of the synthetic functions.
  2. An example for such an object is given in the `v2_news/news_classifier.py` in the context of tuning svm for the 20 news group data-set. The file is fairly self-explanatory and the user can create similar objects for other parameter tuning examples. 
  3. In order to run our example for the given budget run `python MFPDOO/experiments_svm.py` from the root directory of the project. 


  _All the results are saved in `examples/results/`._
