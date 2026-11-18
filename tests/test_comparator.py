from comparator import compare, compare_float, compare_output, compare_unordered, normalize


def test_normalize_line_endings_and_trailing_whitespace():
    assert normalize("1 \r\n2\t\r\n\r\n\n") == "1\n2"


def test_exact_text_match():
    assert compare("hello\n", "hello")
    assert not compare("hello", "Hello")


def test_json_whitespace_insensitive():
    assert compare("[0, 1]\n", "[0,1]")
    assert compare('{"a": [1, 2]}', '{"a":[1,2]}')


def test_json_order_matters():
    assert not compare("[1,0]", "[0,1]")


def test_booleans_are_not_numbers():
    # Regression: Python's True == 1 made "true" accepted for expected "1".
    assert not compare("true", "1")
    assert not compare("[true]", "[1]")
    assert not compare("0", "false")
    assert compare("true", "true")


def test_numbers_compare_by_value():
    assert compare("3", "3.0")
    assert not compare("3", "3.5")


def test_non_json_outputs():
    assert compare("1 2 3", "1 2 3 ")
    assert not compare("1 2 3", "1 2  3")
    assert not compare("", "0")


def test_compare_float_tolerance():
    assert compare_float("0.3333334", "0.3333333")
    assert not compare_float("0.34", "0.33")
    assert compare_float("1000000.1", "1000000.0", tolerance=1e-5)
    assert not compare_float("nan", "1")


def test_compare_unordered_arrays():
    assert compare_unordered("[1,0]", "[0,1]")
    assert compare_unordered('[[1,2],[3]]', '[[3],[1,2]]')
    assert not compare_unordered("[1,1]", "[0,1]")
    assert not compare_unordered("1", "[1]")
    assert not compare_unordered("not json", "[1]")


def test_compare_output_modes():
    assert compare_output("[1,0]", "[0,1]", "EXACT") is False
    assert compare_output("[1,0]", "[0,1]", "UNORDERED") is True
    assert compare_output("0.30000000000000004", "0.3", "FLOAT") is True
    assert compare_output("3", "3.0", "EXACT") is True
    assert compare_output("true", "1", "EXACT") is False
