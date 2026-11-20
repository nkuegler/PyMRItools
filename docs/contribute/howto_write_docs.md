# How to write documentation

This guide explains how the documentation of PyMRItools is organised, how to preview it locally,
and how it is published.

## Documentation structure

The documentation is built with [MkDocs](https://www.mkdocs.org/) using the `readthedocs` theme.
The sources are Markdown files in the `docs/` directory:

| Path | Content |
|---|---|
| `docs/index.md` | Landing page: overview, module map, quick start, references |
| `docs/installation.md`, `docs/concepts.md`, `docs/cli.md` | Installation, shared conventions, list of command line tools |
| `docs/<tool>.md` | One page per tool group (sequence programming, raw data, reconstruction, processing, simulation, modeling) |
| `docs/tutorials/` | Step-by-step workflows |
| `docs/api/` | API reference pages, generated from docstrings |
| `docs/contribute/` | Contribution guidelines |
| `docs/javascripts/`, `docs/stylesheets/` | MathJax configuration and small style adjustments |

## The MkDocs configuration

`mkdocs.yml` in the repository root defines

1. site and repository information,
2. the theme,
3. the Markdown extensions (admonitions, math, tables, …),
4. the navigation (`nav`),
5. the plugins: `search` and `mkdocstrings` for the API reference.

To **add** a page, create the Markdown file in `docs/` and add it to the `nav` section of `mkdocs.yml`.

## API reference from docstrings

The pages in `docs/api/` contain [mkdocstrings](https://mkdocstrings.github.io/) directives such as

```markdown
::: pymritools.recon.loraks.ac_loraks
```

which render the module with all public classes and functions, their signatures, docstrings and source code.
The source is analysed statically, i.e. PyMRItools does not need to be importable to build the documentation.
When you add a new module, add a directive to the matching API page.
Docstrings are parsed in Google style (see [Writing Code](howto_write_code.md)).

## Math

Formulas are written in LaTeX and rendered with MathJax (`pymdownx.arithmatex`):

```markdown
Inline math: \( S(t) = S_0 e^{-R t} \)

Display math:

\[
\alpha = \arccos\left(\frac{r n - 1}{n - r}\right)
\]
```

Inside list items, indent display math by four spaces and surround it with blank lines.

## Building and deploying

The documentation is built and published by the GitHub Actions workflow `.github/workflows/mkdocs.yml`:

1. **Trigger:** a push to the `dev` branch of the main repository.
2. **Setup:** Python is installed together with `mkdocs`, `mkdocstrings` and `mkdocstrings-python`.
3. **Deploy:** `mkdocs gh-deploy --force` builds the site and pushes it to the `gh-pages` branch.
4. GitHub Pages serves the `gh-pages` branch at <https://schmidt-jo.github.io/PyMRItools/>.

All Markdown extensions used here are part of these packages (`pymdown-extensions` is a dependency of `mkdocstrings`),
so no additional packages are needed in the workflow.

## Previewing locally

Install the same packages as the workflow, e.g. in a separate environment:

```shell
pip install mkdocs mkdocstrings mkdocstrings-python
```

or run them with uv in a temporary environment, without touching the project environment:

```shell
uvx --with mkdocstrings-python mkdocs serve
```

Then open <http://127.0.0.1:8000/>; the preview updates when files change.
Before opening a pull request, check that the build has no warnings:

```shell
mkdocs build --strict
```

## Rules for writing

There is a small set of rules that are non-negotiable:

1. Always write in English and use a spell and grammar check.
   In PyCharm you can use the Grazie plugin.
2. Start a new line after the end of each sentence.
   This makes it much easier to see differences in pull requests.
3. Always remember, you are writing documentation for people who don't know how things work.
   Be precise and clear in your writing.
   Your future self will thank you.
4. Document what the code does.
   Option names, defaults, units and file names in the documentation must match the code;
   when you change them in the code, update the documentation in the same pull request.

### Markdown tips

MkDocs supports standard Markdown plus the enabled extensions:

```markdown
# Heading 1
## Heading 2

*Italic text*, **bold text**, `inline code`

- Bullet point
1. Numbered item

[Link text](https://example.com), [link to another page](concepts.md#file-formats)

| Column 1 | Column 2 |
|---|---|
| value | value |

!!! note
    This is an admonition box for notes (also: tip, warning).
```
