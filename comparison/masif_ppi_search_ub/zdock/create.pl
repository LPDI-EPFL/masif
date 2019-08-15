#!/usr/bin/perl

# Rong Chen 6/24/2002
# modified by Brian Pierce 11/22/2010

# This program creates the structural predictions from a zdock output file

use strict;

my $outfile = $ARGV[0];

if ($outfile eq "")
{
    die("usage: create.pl zdock_output_file num_preds\n");
}
my $num_preds = $ARGV[1];
if (($num_preds eq "") || ($num_preds <= 0)) { $num_preds = 100000; }

if (!(-e "./create_lig")) { die("error: need to have create_lig executable linked or copied to current directory\n"); }

# open the zdock output file
open (ZDOUT, "$outfile") || die "Unable to open file $outfile!\n";
my @zdout_lines = <ZDOUT>;
chomp(@zdout_lines);
close(ZDOUT);

# parse the header of the zdock output file
(my $n, my $spacing, my $switch_num)=split(" ", $zdout_lines[0]);

my $line_num = 1;
my $rec_rand1 = 0.0, my $rec_rand2 = 0.0, my $rec_rand3 = 0.0;
if ($switch_num ne "")
{
    ($rec_rand1, $rec_rand2, $rec_rand3)= split(" ", $zdout_lines[$line_num++]);
}
(my $lig_rand1, my $lig_rand2, my $lig_rand3)=split(" ", $zdout_lines[$line_num++]);
(my $rec, my $r1, my $r2, my $r3) = split (" ", $zdout_lines[$line_num++]);
(my $lig, my $l1, my $l2, my $l3) = split (" ", $zdout_lines[$line_num++]);

if ($switch_num eq "1")
{
    my $temp_name = $rec;
    $rec = $lig;
    $lig = $temp_name;
}

# generate the predictions
my $pred_num = 1;
for (my $i = $line_num; ($i < @zdout_lines) && ($pred_num <= $num_preds); $i++)
{
    (my $angl_x, my $angl_y, my $angl_z, my $tran_x, my $tran_y, my $tran_z, my $score) = split ( " ", $zdout_lines[$i] );
    my $newligfile = $outfile . "." . $pred_num;
    my $create_cmd = "./create_lig $newligfile $lig $lig_rand1 $lig_rand2 $lig_rand3 $r1 $r2 $r3 $l1 $l2 $l3 $angl_x $angl_y $angl_z $tran_x $tran_y $tran_z $n $spacing\n";
    if ($switch_num ne "")
    {
	$create_cmd = "./create_lig $newligfile $lig $switch_num $rec_rand1 $rec_rand2 $rec_rand3 $lig_rand1 $lig_rand2 $lig_rand3 $r1 $r2 $r3 $l1 $l2 $l3 $angl_x $angl_y $angl_z $tran_x $tran_y $tran_z $n $spacing\n";
    }

    #print "executing: $create_cmd\n";
    system($create_cmd);
    system "cat $rec $newligfile > complex." . "$pred_num.pdb";
    system "rm $newligfile\n";
    $pred_num++;
}
