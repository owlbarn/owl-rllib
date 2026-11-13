type t =
  | Discrete of int
  | Box of { low : float array; high : float array; shape : int array }

(* Bounds are flattened in row-major order. Observations have no batch axis. *)
let box ~low ~high ~shape =
  let n = Array.fold_left ( * ) 1 shape in
  if Array.length low <> n || Array.length high <> n
     || Array.exists (( >= ) 0) shape
     || not (Array.for_all2 ( <= ) low high)
  then invalid_arg "Space.box: invalid shape or bounds";
  Box { low = Array.copy low; high = Array.copy high; shape = Array.copy shape }
