# -*- coding: utf-8 -*-

import time

# Configuration file for the Sphinx documentation builder.
#
# This file does only contain a selection of the most common options. For a
# full list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

project = u'Ecoevolity model prior experiments'
copyright = u'2020, Jamie Oaks'
author = u'Jamie Oaks'

# The short X.Y version.
version = u'0.1'
# The full version, including alpha/beta/rc tags.
release = u'0.1.0'


# -- General configuration ---------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
# needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
        'sphinx.ext.todo',
        'sphinx.ext.imgmath',
        'sphinx.ext.githubpages',
        'sphinxcontrib.bibtex',
        # 'recommonmark',
        ]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
source_suffix = ['.rst', '.md']
# source_suffix = {
#         '.rst': 'restructeredtext',
#         '.txt': 'markdown',
#         '.md': 'markdown',
#         }

# The master toctree document.
master_doc = 'index'

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = None

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
        '_build',
        'Thumbs.db',
        '.DS_Store',
        'snippets/*.rst',
        ]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'alabaster'

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
html_theme_options = {
        # 'logo': 'ecoevolity-logo-compact.svg',
        'github_user': 'phyletica',
        'github_repo': 'ecoevolity-model-prior',
        'github_button': True,
        'github_banner': False,
        'description': 'Experimenting with new comparative phylogeographical models',
        'show_powered_by': True,
        'fixed_sidebar': False,
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static', '_static/custom.css']

# Custom sidebar templates, must be a dictionary that maps document names
# to template names.
#
# The default sidebars (for documents that don't match any pattern) are
# defined by theme itself.  Builtin themes are using these templates by
# default: ``['localtoc.html', 'relations.html', 'sourcelink.html',
# 'searchbox.html']``.
#
html_sidebars = {
        '**': [
            'about.html',
            'navigation.html',
            'relations.html',
            'searchbox.html',
            'donate.html',
        ]
}


# -- Options for HTMLHelp output ---------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = 'ecoevolity-model-priordoc'


# -- Options for LaTeX output ------------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    # 'papersize': 'letterpaper',

    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',

    # Additional stuff for the LaTeX preamble.
    #
    # 'preamble': '',

    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
    "preamble": r"""
\usepackage{xspace}
\newcommand{\given}{\ensuremath{\,|\,}\xspace}
\newcommand{\pr}{\ensuremath{p}}
\newcommand{\data}{\ensuremath{D}\xspace}
\newcommand{\model}[1][]{\ensuremath{M_{#1}}\xspace}
\newcommand{\parameters}[1][]{\ensuremath{\Theta_{#1}}\xspace}
\newcommand{\parameter}[1][]{\ensuremath{\theta_{#1}}\xspace}
\newcommand{\diff}[1]{\ensuremath{\mathrm{d}#1}}
\newcommand{\ncomparisons}{\ensuremath{\mathcal{N}\xspace}}
\newcommand{\nevents}[1][]{\ensuremath{k_{#1}\xspace}}
\newcommand{\nloci}[1][]{\ensuremath{m_{#1}\xspace}}
\newcommand{\allelecount}[1][]{\ensuremath{\nodeallelecount{#1}{}}\xspace}
\newcommand{\redallelecount}[1][]{\ensuremath{\noderedallelecount{#1}{}}\xspace}
\newcommand{\leafallelecounts}[1][]{\ensuremath{\mathbf{n}_{#1}}\xspace}
\newcommand{\leafredallelecounts}[1][]{\ensuremath{\mathbf{r}_{#1}}\xspace}
\newcommand{\comparisondata}[1][]{\ensuremath{D_{#1}}\xspace}
\newcommand{\alldata}[1][]{\ensuremath{\mathbf{D}}\xspace}
\newcommand{\rgmurate}{\ensuremath{u}\xspace}
\newcommand{\grmurate}{\ensuremath{v}\xspace}
\newcommand{\murate}[1][]{\ensuremath{\mu_{#1}}\xspace}
\newcommand{\murates}[1][]{\ensuremath{\boldsymbol{\mu}_{#1}}\xspace}
\newcommand{\gfreq}[1][]{\ensuremath{\pi_{#1}}\xspace}
\newcommand{\gfreqs}[1][]{\ensuremath{\boldsymbol{\pi}_{#1}}\xspace}
\newcommand{\comparisondivtime}[1][]{\ensuremath{t_{#1}}\xspace}
\newcommand{\comparisondivtimes}[1][]{\ensuremath{\mathbf{t}_{#1}}\xspace}
\newcommand{\divtime}[1][]{\ensuremath{\tau_{#1}}\xspace}
\newcommand{\divtimes}[1][]{\ensuremath{\boldsymbol{\tau}_{#1}}\xspace}
\newcommand{\divtimemodel}[1][]{\ensuremath{T_{#1}}\xspace}
\newcommand{\divtimesets}{\ensuremath{\mathcal{T}}\xspace}
\newcommand{\comparisoneventtime}[1][]{\ensuremath{t_{#1}}\xspace}
\newcommand{\comparisoneventtimes}[1][]{\ensuremath{\mathbf{t}_{#1}}\xspace}
\newcommand{\eventtime}[1][]{\ensuremath{\tau_{#1}}\xspace}
\newcommand{\eventtimes}[1][]{\ensuremath{\boldsymbol{\tau}_{#1}}\xspace}
\newcommand{\eventtimemodel}[1][]{\ensuremath{T_{#1}}\xspace}
\newcommand{\eventtimesets}{\ensuremath{\mathcal{T}}\xspace}
\newcommand{\genetree}[1][]{\ensuremath{g_{#1}}\xspace}
\newcommand{\sptree}[1][]{\ensuremath{S_{#1}}\xspace}
\newcommand{\sptrees}[1][]{\ensuremath{\mathbf{S}_{#1}}\xspace}
\newcommand{\descendantpopindex}[1]{\ensuremath{D{#1}}}
\newcommand{\rootpopindex}[1][]{\ensuremath{R{#1}}\xspace}
\newcommand{\epopsize}[1][]{\ensuremath{N_{e}^{#1}}\xspace}
\newcommand{\comparisonpopsizes}[1][]{\ensuremath{\mathbb{N}_{e}{#1}}\xspace}
\newcommand{\collectionpopsizes}[1][]{\ensuremath{\mathbf{N_{e}}_{#1}}\xspace}
\newcommand{\rootrelativepopsize}{\ensuremath{R_{\epopsize[\rootpopindex]}}\xspace}
\newcommand{\concentration}{\ensuremath{\alpha}\xspace}
\newcommand{\basedistribution}{\ensuremath{H}\xspace}
""",
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (master_doc, 'ecoevolity-model-prior.tex', u'Ecoevolity Model Prior Project Documentation',
     u'Jamie Oaks', 'manual'),
]


# -- Options for manual page output ------------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    (master_doc, 'ecoevolity-model-prior', u'Ecoevolity Model Prior Project Documentation',
     [author], 1)
]


# -- Options for Texinfo output ----------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (master_doc, 'ecoevolity-model-prior', u'Ecoevolity Model Prior Project Documentation',
     author, 'ecoevolity-model-prior', 'Experimenting with new comparative phylogeographical models.',
     'Miscellaneous'),
]


# -- Options for Epub output -------------------------------------------------

# Bibliographic Dublin Core info.
epub_title = project

# The unique identifier of the text. This can be a ISBN number
# or the project homepage.
#
# epub_identifier = ''

# A unique identification for the text.
#
# epub_uid = ''

# A list of files that should not be packed into the epub file.
epub_exclude_files = ['search.html']


# -- Extension configuration -------------------------------------------------

# -- Options for todo extension ----------------------------------------------

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True

imgmath_image_format = "svg"
imgmath_latex_preamble = latex_elements["preamble"]

rst_epilog = """
.. |jro| replace:: Jamie Oaks
.. _jro: http://phyletica.org

.. |phyleticalab| replace:: Phyletica Lab 
.. _phyleticalab: http://phyletica.org

.. |eco| replace:: ecoevolity
.. _eco: http://phyletica.org/ecoevolity
.. |Eco| replace:: Ecoevolity
.. _Eco: http://phyletica.org/ecoevolity
.. |eco_gh| replace:: ecoevolity
.. _eco_gh: https://github.com/phyletica/ecoevolity
.. |eco_url| replace:: http://phyletica.org/ecoevolity
.. |eco_gh_url| replace:: https://github.com/phyletica/ecoevolity
.. |eco_copyright| replace:: **Copyright 2015-{this_year} Jamie R. Oaks**

.. |project_repo| replace:: project repository 
.. _project_repo: https://github.com/phyletica/ecoevolity-model-prior
.. |project_site| replace:: project site 
.. _project_site: http://phyletica.org/ecoevolity-model-prior
.. |project_url| replace:: http://phyletica.org/ecoevolity-model-prior

.. |pyco| replace:: pycoevolity 
.. _pyco: https://github.com/phyletica/pycoevolity
.. |Pyco| replace:: Pycoevolity
.. _Pyco: https://github.com/phyletica/pycoevolity

.. |Anaconda| replace:: Anaconda
.. _Anaconda: https://www.anaconda.com/
.. |Miniconda| replace:: Miniconda
.. _Miniconda: https://docs.conda.io/en/latest/miniconda.html
.. |conda| replace:: ``conda``

.. |simco| replace:: ``simcoevolity``
.. |sumco| replace:: ``sumcoevolity``
.. |dpprobs| replace:: DPprobs
.. |cdpprobs| replace:: ``dpprobs``
.. |ceco| replace:: ``ecoevolity``
.. |pyco-sumchains| replace:: ``pyco-sumchains``
.. |pyco-sumtimes| replace:: ``pyco-sumtimes``
.. |pyco-sumevents| replace:: ``pyco-sumtimes``

.. |Tracer| replace:: Tracer
.. _Tracer: http://tree.bio.ed.ac.uk/software/tracer/

.. |git| replace:: Git
.. _git: http://git-scm.com/

.. |yaml| replace:: YAML 
.. _yaml: http://yaml.org/
.. |yamllint| replace:: http://www.yamllint.com/

.. |gpl3| replace:: http://www.gnu.org/licenses/gpl-3.0-standalone.html

.. |cmake| replace:: CMake
.. _cmake: https://cmake.org/
""".format(this_year = time.strftime('%Y'))
