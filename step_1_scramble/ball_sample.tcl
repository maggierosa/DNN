proc sample_sphere {RAD} {
    set banana 0
    while {$banana==0} {
        set x [expr $RAD*(2*rand()-1.0)]
        set y [expr $RAD*(2*rand()-1.0)]
        set z [expr $RAD*(2*rand()-1.0)]
        set rad_squared [ expr {pow($x, 2) + pow($y, 2) + pow($z, 2)} ]
        set RAD_squared [ expr {pow($RAD, 2)} ]
        if {$rad_squared<=$RAD_squared} {
            set end [list $x $y $z]
            return $end
        }
    }
}
