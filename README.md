# numericsinfsharp

This is the home of my work for the 2021 spring semester, focusing on convex
analysis, measure and probability, and optimization. Written mostly in F#, with
some Python for verification.

## Okay, maybe more than 'some' 

Truth be told, I have almost zero idea what I'm doing with F#. I have some
experience with SMLNJ from an online course I did last summer, which thankfully
means I don't have to *completely* transition into FP, and have a leg up on
starting with F# since it's a member of the 'ML Family.' That said, I will be
*starting* with F# in this project: I have no prior experience. That's why the
tagline says 'some Python for verification.'

I've been doing numerical/scientific programming, particularly with Python, for
years now in my professional life. So, as a twist to the work I'm doing for my
classes, which cover mostly topics I've worked with already, I thought I'd take
this opportunity to learn a new language. However, I'll quite often need to fall
back on Python for things like SymPy, and SciPy.optimize.

## Should Be Fun!

All of the above notwithstanding, the primary purpose of this repo is to catalog
the work I'm doing on learning F#, and to have a convenient place for me to
refer to in the future in the event I find I need a quick way to solve
GradientDescent while hacking away at something else on the CLR. 

## One Final Note

About building this code. Should you, for some absurd reason, feel the need or
desire to want to run any of this code, you have a couple of options. I'm doing
this work on a Win10 machine with Visual Studio, so each assignment or
mini-project will be organized as such. You can also run any of the F# code on
Linux or Mac OSX, via .NET Core, although I leave package management up to you.
Python is python, although I should note that my deps for Python generally
include Python 3 (because its not 2005 anymore), the SciPy stack (SciPy, NumPy,
Pandas and Matplotlib) and SymPy. If you're on Linux or Mac OSX, you have pip,
so those are easily handled. If you're on windows like me, I highly recommend
going with [Anaconda](https://www.anaconda.com/).
