# Creating a conda environment from an environment.yml file

You can create a conda environment from two types of environment files. One of 
the files specifies every installed package, as well as its version numbers. This 
works well when you are creating the environment on the same operating system,
but it works poorly when moving between operating systems (windows to mac, etc.).

A simplified environment file only records the packages the user explicitly 
requests within conda. We'll use that type.

```shell
conda env create --file environment.yml --name deeplearn
```

If successful, conda should install all of it's packages, and then perform a 
final step where it uses pip to install any additional dependencies.

> NOTE: In the case where conda fails to install the packages, then we can look
> through the _environment.yml_ file and install the listed packages manually.
> Just make sure to do the pip installations as the last step.

You can read more information on managing conda environments from the conda 
[documentation](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file)

