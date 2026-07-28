from pathlib import Path


def main() -> None:
    source = Path(__file__).parent / "host" / "g4c_b4_epoch_runner.cpp"
    text = source.read_text(encoding="utf-8")
    start = text.index("int RunPerfBlock(")
    end = text.index("\n}\n}  // namespace", start)
    body = text[start:end]

    assert "std::shared_ptr<ge::Graph> graph;" in body
    assert "ge::Graph graph;" not in body
    assert "session->AddGraph(0, *graph)" in body

    finalizes = []
    offset = 0
    needle = "ge::GEFinalize();"
    while True:
        index = body.find(needle, offset)
        if index < 0:
            break
        finalizes.append(index)
        offset = index + len(needle)
    assert len(finalizes) == 4
    for index in finalizes:
        prefix = body[max(0, index - 100) : index]
        assert prefix.rfind("session.reset();") < prefix.rfind("graph.reset();")

    print("attempt70b-r2-graph-lifetime-selftest\tPASS")


if __name__ == "__main__":
    main()
