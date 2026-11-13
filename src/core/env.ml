(* Tabular environments: the agent can only reset and step, it cannot see the
   model behind them. *)

type step =
  { state : int
  ; reward : float
  ; terminated : bool (* reached a terminal state: do not bootstrap *)
  ; truncated : bool (* hit the step limit: the episode ends, but bootstrap *)
  }

type t =
  { n_states : int
  ; n_actions : int
  ; reset : unit -> int
  ; step : int -> step
  }

(* hide the model of an MDP behind reset / step *)
let of_mdp ?(max_steps = max_int) ~rng (mdp : Mdp.t) =
  let state = ref 0 in
  let steps = ref 0 in
  let reset () =
    state := Mdp.sample_initial rng mdp;
    steps := 0;
    !state
  in
  let step action =
    let tr = Mdp.sample_transition rng mdp !state action in
    state := tr.next;
    incr steps;
    { state = tr.next
    ; reward = tr.reward
    ; terminated = tr.terminal
    ; truncated = (not tr.terminal) && !steps >= max_steps
    }
  in
  { n_states = mdp.n_states; n_actions = mdp.n_actions; reset; step }
