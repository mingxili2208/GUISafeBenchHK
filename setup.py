"""Package metadata for SafeBenchHK."""

from pathlib import Path

from setuptools import find_packages, setup


BASE_DIR = Path(__file__).resolve().parent
README_PATH = BASE_DIR / 'README.md'


setup(
      name='safebench',
      version='1.0.0',
      description='SafeBenchHK: CARLA-based safe and adversarial autonomous driving benchmark',
      long_description=README_PATH.read_text(encoding='utf-8') if README_PATH.exists() else '',
      long_description_content_type='text/markdown',
      packages=find_packages(include=['safebench', 'safebench.*']),
      include_package_data=True,
      python_requires='>=3.8',
      install_requires=['gym', 'pygame'],
)
