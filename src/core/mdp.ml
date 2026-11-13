(* Finite Markov decision processes with an explicit model.

   States and actions are integers. The discount factor is not part of the
   model; it is passed to the algorithms that need it. *)

type transition =
  { prob : float
  ; next : int
  ; reward : float
  ; terminal : bool (* the episode ends on this transition: do not bootstrap *)
  }

type t =
  { n_states : int
  ; n_actions : int
  ; transitions : transition list array array (* transitions.(s).(a) *)
  ; initial : float array (* distribution of the first state *)
  }

(* draw an index from a discrete distribution given as (index, prob) pairs *)
let sample_from rng pairs =
  let u = Random.State.float rng 1. in
  let rec go acc = function
    | [ (x, _) ] -> x
    | (x, p) :: rest -> if u < acc +. p then x else go (acc +. p) rest
    | [] -> invalid_arg "Mdp.sample_from: empty distribution"
  in
  go 0. pairs


let sample_initial rng mdp =
  Array.to_list mdp.initial
  |> List.mapi (fun s p -> s, p)
  |> List.filter (fun (_, p) -> p > 0.)
  |> sample_from rng


let sample_transition rng mdp s a =
  List.map (fun tr -> tr, tr.prob) mdp.transitions.(s).(a) |> sample_from rng
