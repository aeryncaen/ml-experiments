-- OpenMathInstruct-2 (nvidia/OpenMathReasoning).
-- SFT-capable: produces both pt_text and sft_segments.

function extract(record)
    local messages = record.messages
    if not messages or #messages == 0 then
        -- Fallback: try problem/solution fields.
        if record.problem and record.solution then
            messages = {
                {role = "user", content = record.problem},
                {role = "assistant", content = record.solution},
            }
        else
            return nil
        end
    end

    -- PT format: concatenated flat text.
    local parts = {}
    for _, msg in ipairs(messages) do
        if msg.content and #msg.content > 0 then
            table.insert(parts, msg.content)
        end
    end
    if #parts == 0 then return nil end

    -- SFT format: structured segments with loss flags.
    local sft_segments = {}
    for _, msg in ipairs(messages) do
        table.insert(sft_segments, {
            role = msg.role or "user",
            content = msg.content or "",
            loss = (msg.role == "assistant"),
        })
    end

    return {
        pt_text = table.concat(parts, "\n\n"),
        sft_segments = sft_segments,
        route = "train",
    }
end
