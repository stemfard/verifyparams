## verifyparams

A Python library for validating and verifying function parameters, arrays, matrices, dataframes, and other input types in scientific and engineering computations.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the verifyparams library.

```bash
pip install verifyparams
```

## Usage

The library is imported into a Python session by running the following import statement.

```python
>>> import verifyparams as vp
```

Example usage:

```python
>>> import verifyparams as vp
```

```python
# Validate numeric input
>>> verify_numeric(param_name='x', value=10)
```

```python
# Validate array/matrix
>>> verify_array_matrix(A, nrows=3, ncols=3, is_square=True)
```

```python
# Validate function
>>> verify_function(f, variable=['x'])
```

## Support

For any support on any of the functions in this library, send us an email at: `verifyparams@stemfard.org`. We are happy to offer assistance wherever possible.

## Roadmap

Future releases aim to make `verifyparams` the go-to library for developers, students, trainers, and professionals who want reliable parameter validation in scientific, engineering, and data-intensive Python applications.

## Contributing

To make `verifyparams` a successful library while keeping the code clean and maintainable, we welcome any valuable contributions towards the development and improvement of this library.

For major changes to the library, please open an issue with us first to discuss what you would like to change and we will be happy to review and implement them.

## Authors and Acknowledgement

We are grateful to the incredible support from our developers at `STEM Research`.

## License

[MIT](https://choosealicense.com/licenses/mit/)