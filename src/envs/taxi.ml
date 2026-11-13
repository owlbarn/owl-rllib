(* Taxi, same dynamics as Gymnasium's Taxi-v3.

   A taxi on a 5x5 grid picks up a passenger at one of four locations and
   drops them at another. Every step costs 1, a successful drop-off earns 20,
   an illegal pick-up or drop-off costs 10. *)

let map =
  [| "+---------+"
   ; "|R: | : :G|"
   ; "| : | : : |"
   ; "| : : : : |"
   ; "| | : | : |"
   ; "|Y| : |B: |"
   ; "+---------+"
  |]


(* pick-up and drop-off locations R, G, Y, B as (row, col) *)
let locs = [| 0, 0; 0, 4; 4, 0; 4, 3 |]

(* passenger: 0-3 waiting at that location, 4 in the taxi *)
let encode row col passenger dest = (((((row * 5) + col) * 5) + passenger) * 4) + dest

let decode s = s / 100, s / 20 mod 5, s / 4 mod 5, s mod 4

(* actions: 0 south, 1 north, 2 east, 3 west, 4 pick up, 5 drop off *)
let make () : Mdp.t =
  let transition s a =
    let row, col, passenger, dest = decode s in
    let here = row, col in
    let at_loc = Array.exists (( = ) here) locs in
    let state ?(row = row) ?(col = col) ?(passenger = passenger) () = encode row col passenger dest in
    let next, reward, terminal =
      match a with
      | 0 -> state ~row:(min (row + 1) 4) (), -1., false
      | 1 -> state ~row:(max (row - 1) 0) (), -1., false
      | 2 when map.(1 + row).[(2 * col) + 2] = ':' -> state ~col:(col + 1) (), -1., false
      | 3 when map.(1 + row).[2 * col] = ':' -> state ~col:(col - 1) (), -1., false
      | 4 when passenger < 4 && here = locs.(passenger) -> state ~passenger:4 (), -1., false
      | 5 when passenger = 4 && here = locs.(dest) -> state ~passenger:dest (), 20., true
      | 5 when passenger = 4 && at_loc ->
        let i = ref 0 in
        while locs.(!i) <> here do
          incr i
        done;
        state ~passenger:!i (), -1., false
      | 4 | 5 -> s, -10., false
      | _ -> s, -1., false
    in
    [ { Mdp.prob = 1.; next; reward; terminal } ]
  in
  let valid_start s =
    let _, _, passenger, dest = decode s in
    passenger < 4 && passenger <> dest
  in
  let initial = Array.init 500 (fun s -> if valid_start s then 1. /. 300. else 0.) in
  { n_states = 500
  ; n_actions = 6
  ; transitions = Array.init 500 (fun s -> Array.init 6 (transition s))
  ; initial
  }
