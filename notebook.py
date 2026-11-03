# /// script
# dependencies = ["marimo"]
# requires-python = ">=3.11"
# ///

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    slider = mo.ui.slider(1,10,1)
    slider 



    return (slider,)


@app.cell
def _(slider):
    slider.value * "hello" 
    return


if __name__ == "__main__":
    app.run()
