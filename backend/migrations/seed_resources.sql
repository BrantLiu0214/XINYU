-- Seed crisis hotline resources for XinYu thesis demo
-- Run once from backend/ directory:
--   psql -U postgres -d xinyu -f migrations/seed_resources.sql

INSERT INTO resource_catalog (id, title, description, phone, link_url, risk_level, is_active)
VALUES
  (gen_random_uuid()::text, '北京心理危机研究与干预中心', '24小时心理援助热线，专业危机干预', '010-82951332', NULL, 'L2', true),
  (gen_random_uuid()::text, '全国心理援助热线', '工作日 9:00–21:00，免费心理援助', '400-161-9995', NULL, 'L2', true),
  (gen_random_uuid()::text, '生命热线', '24小时危机干预，保密免费', '400-821-1215', NULL, 'L3', true),
  (gen_random_uuid()::text, '希望24热线', '24小时公益心理援助与危机干预', '400-161-9995', 'http://www.hope24line.com', 'L3', true)
ON CONFLICT DO NOTHING;
