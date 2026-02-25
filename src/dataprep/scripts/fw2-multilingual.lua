-- FineWeb2 multilingual datasets.
-- Shared by all fw2-* datasets. Quality filter on length and URL density.

function extract(record)
    local text = record.text
    if not text then return nil end

    -- Quality filter: skip very short docs.
    if utf8_len(text) < 100 then
        return nil
    end

    -- Skip docs that are mostly URLs/code.
    local url_count = 0
    for _ in text:gmatch("https?://") do
        url_count = url_count + 1
    end
    if url_count > 20 then
        return nil
    end

    return {
        pt_text = text,
        route = "train",
    }
end
