(** Model-free prediction and control with tables (Sutton and Barto, chapters 5 and 6). A policy is
    a function from state to action; it may sample internally. *)

val argmax : Random.State.t -> float array -> int
(** An index of the largest element, ties broken at random. *)

val epsilon_greedy : Random.State.t -> epsilon:float -> float array -> int

val mc_prediction : gamma:float -> episodes:int -> policy:(int -> int) -> Env.t -> float array
(** First-visit Monte Carlo prediction. *)

val td0_prediction :
  gamma:float -> alpha:float -> episodes:int -> policy:(int -> int) -> Env.t -> float array

val sarsa :
  Random.State.t ->
  gamma:float ->
  alpha:float ->
  epsilon:float ->
  episodes:int ->
  Env.t ->
  float array array * float array
(** On-policy TD control with an epsilon-greedy policy. Returns the action values and the
    undiscounted return of every episode. *)

val q_learning :
  Random.State.t ->
  gamma:float ->
  alpha:float ->
  epsilon:float ->
  episodes:int ->
  Env.t ->
  float array array * float array
(** Off-policy TD control: bootstrap from the best next action. *)
