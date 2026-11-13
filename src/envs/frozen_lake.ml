(* FrozenLake, same dynamics as Gymnasium's FrozenLake-v1.

   'S' start, 'F' frozen, 'H' hole (terminal), 'G' goal (terminal, reward 1).
   On slippery ice the agent moves in the intended direction or in one of the
   two perpendicular directions, each with probability 1/3. *)

let map_4x4 = [| "SFFF"; "FHFH"; "FFFH"; "HFFG" |]

(* actions: 0 left, 1 down, 2 right, 3 up *)
let moves = [| 0, -1; 1, 0; 0, 1; -1, 0 |]

let make ?(slippery = true) ?(map = map_4x4) () : Mdp.t =
  let rows = Array.length map
  and cols = String.length map.(0) in
  let cell s = map.(s / cols).[s mod cols] in
  let terminal s = cell s = 'G' || cell s = 'H' in
  let move s a =
    let dr, dc = moves.(a) in
    let r = max 0 (min (rows - 1) ((s / cols) + dr))
    and c = max 0 (min (cols - 1) ((s mod cols) + dc)) in
    (r * cols) + c
  in
  let outcome prob s a =
    let next = move s a in
    { Mdp.prob; next; reward = (if cell next = 'G' then 1. else 0.); terminal = terminal next }
  in
  let transitions =
    Array.init (rows * cols) (fun s ->
        Array.init 4 (fun a ->
            if terminal s
            then [ { Mdp.prob = 1.; next = s; reward = 0.; terminal = true } ]
            else if slippery
            then List.map (outcome (1. /. 3.) s) [ (a + 3) mod 4; a; (a + 1) mod 4 ]
            else [ outcome 1. s a ]))
  in
  let starts = List.filter (fun s -> cell s = 'S') (List.init (rows * cols) Fun.id) in
  let p = 1. /. float_of_int (List.length starts) in
  let initial = Array.init (rows * cols) (fun s -> if List.mem s starts then p else 0.) in
  { n_states = rows * cols; n_actions = 4; transitions; initial }
