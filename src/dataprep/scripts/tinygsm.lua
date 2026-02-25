-- TinyGSM datasets (dolmino-mix-1124 tinygsm-* subsets).
-- SFT-capable: math problem solving.

function extract(record)
    local text = record.text
    if not text or #text < 20 then
        return nil
    end

    -- Try to split on common Q/A separators.
    local question, answer

    -- Look for "####" separator (GSM8K style).
    local sep_pos = text:find("####")
    if sep_pos then
        question = text:sub(1, sep_pos - 1)
        answer = text:sub(sep_pos)
    end

    if question and answer and #question > 10 and #answer > 5 then
        return {
            pt_text = text,
            sft_segments = {
                {role = "user", content = trim(question), loss = false},
                {role = "assistant", content = trim(answer), loss = true},
            },
            route = "train",
        }
    end

    -- Fallback: just PT, no SFT.
    return {
        pt_text = text,
        route = "train",
    }
end
