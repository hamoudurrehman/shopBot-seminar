-- Run this in your Supabase SQL Editor (Dashboard -> SQL Editor -> New Query):

CREATE TABLE IF NOT EXISTS public.user_memories (
    user_id TEXT NOT NULL,
    pref_key TEXT NOT NULL,
    pref_value TEXT NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL,
    PRIMARY KEY (user_id, pref_key)
);

-- Grant full table permissions to anon, authenticated, and service_role
GRANT ALL ON TABLE public.user_memories TO postgres, anon, authenticated, service_role;

-- Enable Row Level Security (RLS) but allow all operations
ALTER TABLE public.user_memories ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "Allow all access to user_memories" ON public.user_memories;

CREATE POLICY "Allow all access to user_memories"
ON public.user_memories
FOR ALL
TO public
USING (true)
WITH CHECK (true);

