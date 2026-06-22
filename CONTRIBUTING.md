# Contributing to VectorGraph-Engine

First off, thank you for considering contributing to VectorGraph-Engine! It's people like you that make VectorGraph-Engine such a great tool.

## Where do I go from here?

If you've noticed a bug or have a feature request, make one! It's generally best if you get confirmation of your bug or approval for your feature request this way before starting to code.

## Fork & create a branch

If this is something you think you can fix, then fork VectorGraph-Engine and create a branch with a descriptive name.

A good branch name would be (where issue #325 is the ticket you're working on):

```sh
git checkout -b 325-add-new-feature
```

## Setup environment

Please ensure you have all dependencies installed as specified in the `README.md`.

```sh
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/macOS
# source venv/bin/activate
pip install -r requirements.txt
```

Make sure you have system-level dependencies (`tesseract`, `poppler`) installed as well.

## Implement your fix or feature

At this point, you're ready to make your changes. Feel free to ask for help; everyone is a beginner at first.

* Please follow Python PEP 8 styling conventions.
* Keep your commits atomic and write descriptive commit messages.

## Running the application locally

You can test your changes by running the Streamlit app locally:

```sh
streamlit run app.py
```

## Make a Pull Request

At this point, you should switch back to your master branch and make sure it's up to date with VectorGraph-Engine's master branch:

```sh
git remote add upstream https://github.com/TusharSaini999/VectorGraph-Engine.git
git checkout master
git pull upstream master
```

Then update your feature branch from your local copy of master, and push it!

```sh
git checkout 325-add-new-feature
git rebase master
git push --set-upstream origin 325-add-new-feature
```

Finally, go to GitHub and make a Pull Request.

## Keeping your Pull Request updated

If a maintainer asks you to "rebase" your PR, they're saying that a lot of code has changed, and that you need to update your branch so it's easier to merge.

Thank you for contributing!
