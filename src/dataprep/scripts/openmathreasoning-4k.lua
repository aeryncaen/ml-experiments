-- OpenMathReasoning (nvidia/OpenMathReasoning, no subset).
-- SFT-capable: structured reasoning traces.

function extract(record)
    local messages = record.messages
    if not messages or #messages == 0 then
        if record.question and record.answer then
            messages = {
                {role = "user", content = record.question},
                {role = "assistant", content = record.answer},
            }
        else
            return nil
        end
    end

    local parts = {}
    local sft_segments = {}
    for _, msg in ipairs(messages) do
        if msg.content and #msg.content > 0 then
            table.insert(parts, msg.content)
            table.insert(sft_segments, {
                role = msg.role or "user",
                content = msg.content or "",
                loss = (msg.role == "assistant"),
            })
        end
    end
    if #parts == 0 then return nil end

    return {
        pt_text = table.concat(parts, "\n\n"),
        sft_segments = sft_segments,
        route = "train",
    }
end
