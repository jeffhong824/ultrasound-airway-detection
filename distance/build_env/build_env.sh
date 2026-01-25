conda env create -f environment.yml &&
conda activate portable &&
pip install ./gcvspline
# git clone https://github.com/charlesll/gcvspline.git
# modify pyproject.toml second line:
#   requires = ["setuptools>=44.0,<=72.2.0", "numpy>=1.12,<1.23.0"]
#
# numpy.distutils has been deprecated in NumPy 1.23.0 (https://numpy.org/devdocs/reference/distutils_status_migration.html)