-- StarCoder / code datasets.
-- Text column is "content" not "text".

function extract(record)
    local content = record.content
    if not content or #content < 50 then
        return nil
    end
    -- Skip mega-files (>1MB of text).
    if #content > 1000000 then
        return nil
    end
    return {
        pt_text = content,
        route = "train",
    }
end
