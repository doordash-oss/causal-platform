# Experimentation Library 

We are thrilled you decided to contribute to Doordash Experiment Library!

## Doordash Experimentation Library Dev Environment setup

#### Python

The easiest way to install the python version for this project is to use `pyenv`. Follow these steps:
* `brew install pyenv`
* Add pyenv initializer to shell startup script `~/.bash_profile`: `echo 'eval "$(pyenv init -)"' >> ~/.bash_profile`
* `pyenv install 3.8.10`
* `pyenv shell 3.8.10`

To confirm that you have the right python version, simply run `python` in your terminal. This codebase should work with any Python ~3.8 version

#### Package dependencies
To install package depenencies, follow these steps:
* `make install-deps`. This command will do the following:
    - It wil create a virtual environment in the root of the project.
    - It will install poetry, which is being used for dependency management and package development
    - Poetry will install all the dependencies from `poetry.lock` file
    - It will install pre-commit hooks that are used for linting and formatting.
* Set up artifactory config by running `poetry config http-basic.artifactory username password` with your artifactory username and password
* If you want to update your dependencies, you can run `poetry update`.

#### Other make commands
* `make shell`: will start a bash terminal inside the container based of `Dockerfile` found in the project directory. This can be useful for running code in a more isolated environment that mimicks the CI/CD system.
* `make local-build`: this will build the sdist and the wheel for the library and put them in `dist` directory.
* `make unittest`: this will run tests locally

To perform development in a container using VsCode, please follow this [guide](https://code.visualstudio.com/docs/remote/containers).


## License
This library is released under the Apache 2.0 license. See [LICENSE](https://github.com/doordash-oss/casual-platform/blob/main/LICENSE.txt) for details.