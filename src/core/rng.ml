(* Fixed stream identifiers keep evaluation and network construction from
   consuming training randomness. These streams are reproducible, not a claim
   of mathematically independent substreams. *)
type t =
  { env : Random.State.t
  ; action : Random.State.t
  ; sample : Random.State.t
  ; eval : Random.State.t
  ; network_seed : int
  }

let create seed =
  let stream i = Random.State.make [| seed; i |] in
  { env = stream 0; action = stream 1; sample = stream 2; eval = stream 3
  ; network_seed = Random.State.bits (stream 4) }

let init_network t = Owl_stats_prng.init t.network_seed

(* Box-Muller, without a hidden cached sample or global random state. *)
let normal rng =
  sqrt (-2. *. log (1. -. Random.State.float rng 1.))
  *. cos (2. *. Float.pi *. Random.State.float rng 1.)
