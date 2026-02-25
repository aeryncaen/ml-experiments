-- Default extraction script.
-- Used for datasets that don't need custom logic.
-- Expects record.text to be present.

function extract(record)
    local text = record.text
    if not text or #text == 0 then
        return nil
    end
    return {
        pt_text = text,
        route = "train",
    }
end
