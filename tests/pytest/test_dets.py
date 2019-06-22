#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import forte

def det(s):
    d = forte.Determinant();
    for k, c in enumerate(s):
        if c == '+':
            d.create_alfa_bit(k)
        elif c == '-':
            d.create_beta_bit(k)
        elif c == '2':
            d.create_alfa_bit(k)
            d.create_beta_bit(k)
    return d

def test_dets_equality():
    """Test the __eq__ operator"""
    print("Testing determinant interface")
    d1 = det("22")
    d2 = det("2+")
    d3 = det("22")
    d4 = det("0022")
    assert d1 == d1
    assert d1 != d2
    assert d1 == d3
    assert d2 != d4
    assert d1 != d4

def test_dets_hash():
    """Test the __hash__ operator"""
    d1 = det("22")
    d2 = det("2+")
    d3 = det("22")
    d4 = det("0022")
    h = {}
    h[d1] = 1.0
    h[d2] = 2.0
    h[d3] += 0.25
    h[d4] = 3.0
    assert h[d1] == 1.25
    assert h[d3] == 1.25
    assert h[d2] == 2.00
    assert h[d4] == 3.00

def test_dets_sorting():
    """Test the __lt__ operator"""
    d1 = det("22")
    d2 = det("2+")
    d3 = det("--")
    d4 = det("22")
    list = [d1, d2, d3, d4]
    print(list)
    sorted_list = sorted(list)
    print(sorted_list)
    assert sorted_list[0] == d2
    assert sorted_list[1] == d3
    assert sorted_list[2] == d1
    assert sorted_list[3] == d4
    assert sorted_list[2] == d4
    assert sorted_list[3] == d1

def test_dets_sorting():
    """Test the gen_operator function"""
    # test a -> a excitation
    d1 = det("220")
    assert d1.gen_excitation([0],[3],[],[]) == -1.0
    assert d1 == det("-20+")

    # test b -> b excitation
    d2 = det("2-+0")
    assert d2.gen_excitation([],[],[0,1],[2,3]) == -1.0
    assert d2 == det("+02-")

    # test b creation and counting number of a
    d3 = det("+000")
    assert d3.gen_excitation([],[],[],[0]) == -1.0
    assert d3 == det("2")
    d3 = det("0000")
    assert d3.gen_excitation([],[],[],[0]) == +1.0
    assert d3 == det("-")

    # test ab creation and sign
    d4 = det("000")
    assert d4.gen_excitation([],[2,1],[],[0,1]) == -1.0
    assert d4 == det("-2+")
    d5 = det("000")
    assert d5.gen_excitation([],[2,1],[],[1,0]) == +1.0
    assert d5 == det("-2+")
    d6 = det("000")
    assert d6.gen_excitation([],[1,2],[],[0,1]) == +1.0
    assert d6 == det("-2+")
    d7 = det("000")
    assert d7.gen_excitation([],[1,2],[],[1,0]) == -1.0
    assert d7 == det("-2+")