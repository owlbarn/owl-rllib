(lang dune 3.16)

(name owl-rllib)

(generate_opam_files true)

(source
 (github owlbarn/owl-rllib))

(license MIT)

(authors "Jianxin Zhao")

(maintainers "Jianxin Zhao")

(package
 (name owl-rllib)
 (synopsis "Reinforcement learning on top of the Owl numerical library")
 (description
  "Tabular methods, dynamic programming, DQN, REINFORCE, A2C and PPO on Owl's\nneural networks, classic control environments checked against Gymnasium,\nRL algorithms for language-model fine-tuning (REINFORCE, RLOO, PPO, GRPO and\nits variants) on a toy model, and an optional bridge to Gymnasium and VizDoom\nrunning in a Python child process.")
 (tags (reinforcement-learning machine-learning owl))
 (depends
  (ocaml (>= 5.1))
  (owl (>= 1.3))
  (alcotest (and :with-test (>= 1.0.0)))))
