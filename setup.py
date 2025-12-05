from setuptools import find_packages, setup


setup(
    name="introduction-deep-learning-dqn",
    version="0.1.2",
    description="Install dependencies for the DQN introduction project.",
    python_requires=">=3.13",
    install_requires=[
        "torch",
        "gymnasium",
        "tensorboard",
    ],
)
