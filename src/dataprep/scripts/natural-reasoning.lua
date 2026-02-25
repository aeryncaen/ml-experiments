-- Natural Reasoning (facebook/natural_reasoning).
-- SFT-capable: question + reasoning answer.

function extract(record)
    local question = record.question or record.problem
    local answer = record.answer or record.solution or record.text

    if not question or not answer then
        -- Fallback: try messages format.
        if record.messages and #record.messages >= 2 then
            question = record.messages[1].content
            answer = record.messages[#record.messages].content
        end
    end

    if not question or not answer then return nil end
    if #question < 10 or #answer < 10 then return nil end

    return {
        pt_text = question .. "\n\n" .. answer,
        sft_segments = {
            {role = "user", content = question, loss = false},
            {role = "assistant", content = answer, loss = true},
        },
        route = "train",
    }
end
