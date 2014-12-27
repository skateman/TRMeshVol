# TRMeshVol

**This is not a serious project, just a solution of a university homework!**

Volume calculation of a model represented by triangular meshes, implemented in CUDA.
The code was implemented and tested on the [Anselm Supercomputer](http://www.it4i.cz/?lang=en) which has nVidia Tesla K20M cards.

### Usage
Compile the code with the `make` command and run the binary from one of the subfolders:

```bash
./volume <N> <file>
```

Where *N* is the number of threads per block (has to be a multiply of 32) and *file* is the path to a [STL](http://en.wikipedia.org/wiki/STL_%28file_format%29) file.

Alternatively you can run a benchmark with the `make test` command, which generates a *report.txt* file.

### Implementations

|                            |  Reduction            |  Data structure       |  Used memory         |  Reduction between blocks       |
|:--------------------------:|:---------------------:|:---------------------:|:--------------------:|:-------------------------------:|
|  simple_aos_global_seqsum  |  sequential indexing  |  array of structures  |  global only         |  sequential using CPU           |
|  simple_soa_global_seqsum  |  sequential indexing  |  structure of arrays  |  global only         |  sequential using CPU           |
|  simple_soa_shared_seqsum  |  sequential indexing  |  structure of arrays  |  shared and global   |  sequential using CPU           |
|  unroll_soa_global_seqsum  |  completely unrolled  |  structure of arrays  |  global only         |  sequential using CPU           |
|  unroll_soa_global_parsum  |  completely unrolled  |  structure of arrays  |  global only         |  parallell using second kernel  |
|  unroll_soa_shared_seqsum  |  completely unrolled  |  structure of arrays  |  shared and global   |  sequential using CPU           |
|  unroll_soa_shared_parsum  |  completely unrolled  |  structure of arrays  |  shared and global   |  parallell using second kernel  |

### Authors
- [Adam Široký](https://github.com/xsirok07)
- [Dávid Halász](https://github.com/skateman)
- [Petr Huták](https://github.com/xhutak00)
