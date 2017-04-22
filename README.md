# nash_network

Source code from paper (AIStat 2017):
https://arxiv.org/abs/1606.08718


bibtex:
@inproceedings{perolat2016learning,
  title={Learning Nash Equilibrium for General-Sum Markov Games from Batch Data},
  author={P{\'e}rolat, Julien and Strub, Florian and Piot, Bilal and Pietquin, Olivier},
  booktitle={Artificial Intelligence and Statistics},
  year={2017}
}

Requirement:
 - tensorflow > 1.0
 - numpy
 - pickle


Given a random generated markov games with N players, this code computes the players'stategie that would tend to an epsilon nash-equilibrium.
Parameters are defined in the main.py for a two plyer general sum-games.

We are aware that the code is not very clean (bad design paterns, hardcoded conf, few comments), I would be happy to answer your questions and update the code accordingly. Feel free to contact me for more details!



 

