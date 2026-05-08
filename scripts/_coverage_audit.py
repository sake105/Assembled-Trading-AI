"""Coverage audit: maps each backlog item to test classes."""

with open("tests/test_session_2026_05_07_new_items.py", encoding="utf-8") as f:
    content = f.read()

classes = set()
for line in content.splitlines():
    if line.startswith("class Test"):
        classes.add(line.strip().rstrip(":"))


def has_match(keywords):
    return any(any(kw.lower() in cls.lower() for kw in keywords) for cls in classes)


# (item_nr, keywords, reason_if_personal)
items = [
    # BACKLOG_NACH_PILOT 1-105
    (1, ["ZeroDivision", "SafeDivide"], ""),
    (2, ["RegimeCache"], ""),
    (3, ["HMMCache"], ""),
    (4, ["KillSwitch", "CompositeScore"], ""),
    (5, ["ExceptException", "ExceptPattern"], ""),
    (6, ["DDDamper"], ""),
    (7, ["DatetimeTZ", "DatetimeTimezone"], ""),
    (8, ["SQLInjection"], ""),
    (9, ["LargeModule"], ""),
    (10, ["NoDuplicate", "Duplicate"], ""),
    (11, ["TypeHint"], ""),
    (12, ["LazyImport"], ""),
    (13, ["RandomSeed", "RandomState"], ""),
    (14, ["FileLock", "DDDamper"], ""),
    (15, ["MagicNumber"], ""),
    (16, ["Iterrows"], ""),
    (17, ["AssertInProd", "ProductionAssert"], ""),
    (18, ["RSSFetcher"], ""),
    (19, ["DependencyPin"], ""),
    (20, ["SecurityAudit", "SecurityScanning", "SecurityTools"], ""),
    (21, ["EnvExample"], ""),
    (22, ["EnvValidation"], ""),
    (23, ["ConfigFile"], ""),
    (24, ["ModelVersion"], ""),
    (25, ["Slippage"], ""),
    (26, ["DataFreshness", "Freshness"], ""),
    (27, ["MemoryProfile"], ""),
    (28, ["SpeculationFrist", "TaxAware"], ""),
    (29, ["PerformanceAttribution"], ""),
    (30, ["Benchmark"], ""),
    (31, ["BacktestNoLeverage", "Backtest2023"], ""),
    (32, ["DailyReview"], ""),
    (33, ["DisasterRunbook", "DisasterRecovery"], ""),
    (34, ["BrokerChoice", "Brokerage"], ""),
    (35, ["PytestSkip"], ""),
    (36, ["DocumentationDrift"], ""),
    (37, ["DocumentationHierarchy"], ""),
    (38, ["ReadmeForPilot", "READMEPilot"], ""),
    (39, ["DecisionLog", "DecisionLogger"], ""),
    (40, ["Onboarding"], ""),
    (41, ["DecimalMoney"], ""),
    (42, ["MarginCall"], ""),
    (43, ["HaltCheck"], ""),
    (44, ["FIFO"], ""),
    (45, ["SpeculationFrist"], ""),
    (46, ["BorrowRate"], ""),
    (47, ["NaNPropagation", "SafeDivide"], ""),
    (48, ["NaNPropagation"], ""),
    (49, ["RollingWindow", "RollingMinPeriods"], ""),
    (50, ["ExceptPattern"], ""),
    (51, ["Iterrows"], ""),
    (52, ["LazyImports"], ""),
    (53, ["TypeAnnotation"], ""),
    (54, ["RandomState"], ""),
    (55, ["ProductionAssert"], ""),
    (56, ["PrePostMarket"], ""),
    (57, ["ETFTracking"], ""),
    (58, ["Spinoff", "SpinOff"], ""),
    (59, ["WashSale"], ""),
    (60, ["MLDrift"], ""),
    (61, ["Retraining"], ""),
    (62, ["FeatureImportance"], ""),
    (63, ["Calibration"], ""),
    (64, ["LoggingHot"], ""),
    (65, ["StructuredLog"], ""),
    (66, ["FileLocking"], ""),
    (67, ["DST"], ""),
    (68, ["PositionState"], ""),
    (69, ["BuyingPower"], ""),
    (70, ["PDT"], ""),
    (71, ["StorageRotation", "CleanupScript"], ""),
    (72, ["DatabaseBackup", "BackupScript"], ""),
    (73, ["MLModelVersion", "ModelRegistry"], ""),
    (74, ["ModelHash"], ""),
    (75, ["BacktestRepro"], ""),
    (76, ["TrailingStop"], ""),
    (77, ["ATR"], ""),
    (78, ["LimitOrder"], ""),
    (79, ["SpreadCapture"], ""),
    (80, ["StaleOrder"], ""),
    (81, ["PreEarnings", "Earnings"], ""),
    (82, ["FOMC"], ""),
    (83, ["ExDividend"], ""),
    (84, ["QuarterEnd"], ""),
    (85, ["MAAnnounc", "MAExclusion"], ""),
    (86, ["BacktestLiveParity", "BacktestPaperParity"], ""),
    (87, ["ForwardTest"], ""),
    (88, ["CPCV"], ""),
    (89, ["WalkForward"], ""),
    (90, ["SettingWithCopy", "PandasChained", "PandasCopy"], ""),
    (91, ["MemoryProfiling"], ""),
    (92, ["PickleSecurity", "SafePickle", "PickleLoading"], ""),
    (93, ["CSVTimezone"], ""),
    (94, ["AllExports", "PublicAPIExports"], ""),
    (95, ["PyTyped"], ""),
    (96, ["FatFinger"], ""),
    (97, ["F821"], ""),
    (98, ["F401", "NoqaF401"], ""),
    (99, ["CIWindows"], ""),
    (100, ["OsPath", "Pathlib"], ""),
    (101, ["DatetimeFormat"], ""),
    (102, ["AuditTrail"], ""),
    (103, ["DecisionLog"], ""),
    (104, ["NoqaDistribution", "NoqaPerFile"], ""),
    (105, ["MarketHoursPolicy", "EnforceMarketHours"], ""),
    # BACKLOG_ERGAENZUNG 106-172
    (106, ["OptionsIV"], ""),
    (107, ["InsiderCluster"], ""),
    (108, ["BuybackDrift"], ""),
    (109, ["PEADSUE"], ""),
    (110, ["HRPWeights"], ""),
    (111, ["Tier1Wiring", "Tier1Modules", "Tier1Remaining"], ""),
    (112, ["FeatureStore"], ""),
    (113, ["UniversePIT"], ""),
    (114, ["SectorBias"], ""),
    (115, ["BacktestNoLeverage"], ""),
    (116, ["Backtest2023"], ""),
    (117, ["EDCLConviction"], ""),
    (118, ["StressTestWithLeverage"], ""),
    (119, ["PilotConfig"], ""),
    (120, ["PilotManifest"], ""),
    (121, ["ModuleGlobal"], ""),
    (122, ["PhantomModule"], ""),
    (123, ["VPS"], ""),
    (124, ["Scheduler"], ""),
    (125, ["BrokerChoice", "Brokerage"], ""),
    (126, [], "PERSONAL — Sunk-Cost-Bias"),
    (127, [], "PERSONAL — Peer-Review-Mentor"),
    (128, [], "STRATEGIC — QuantConnect-Migration"),
    (129, [], "PERSONAL — Was wenn Pilot gewinnt"),
    (130, [], "PERSONAL — Lebens-Sustainability 16h"),
    (131, ["PilotLearnings", "PilotV1Crash"], ""),
    (132, ["PilotV2Init"], ""),
    (133, ["ExposureCeiling", "MaxExposureMult"], ""),
    (134, ["DataSourceFallback", "YFinanceFallback"], ""),
    (135, ["SingleMachine"], ""),
    (136, ["ABCompare"], ""),
    (137, ["RequirementsLock"], ""),
    (138, ["PreCommit"], ""),
    (139, ["CommentLanguage"], ""),
    (140, ["NoqaDistribution"], ""),
    (141, ["VariableNaming"], ""),
    (142, ["ModuleDocstring"], ""),
    (143, ["UnusedImport"], ""),
    (144, ["FormatConsistency"], ""),
    (145, ["AsyncNewsFetcher"], ""),
    (146, [], "PERSONAL — Lebens-Trading-Plan"),
    (147, ["DailyReview"], ""),
    (148, ["PilotSuccess"], ""),
    (149, ["KnownIssues"], ""),
    (150, ["Bandit"], ""),
    (151, ["TokenHandling"], ""),
    (152, ["NewsAPIRate"], ""),
    (153, ["MLModelPolicy"], ""),
    (154, ["MLModelPolicy"], ""),
    (155, ["Tier1ModulesPresent"], ""),
    (156, ["ConfigDir"], ""),
    (157, ["TradingCycle"], ""),
    (158, ["ExceptPatternBound"], ""),
    (159, ["TradingCycle"], ""),
    (160, ["WebCrawler"], ""),
    (161, ["TestResource"], ""),
    (162, ["LoggingRotation", "LogRotation"], ""),
    (163, ["DisasterRecovery"], ""),
    (164, ["NetworkTimeout"], ""),
    (165, ["EDGAR"], ""),
    (166, ["Brokerage"], ""),
    (167, ["TaxAware"], ""),
    (168, ["DrawdownPsych"], ""),
    (169, ["BacklogDoc"], ""),
    (170, ["BacklogDoc"], ""),
    (171, ["BacklogDoc"], ""),
    (172, ["BacklogDoc"], ""),
]

covered = []
uncovered = []
personal = []

for item_nr, keywords, reason in items:
    if reason:
        personal.append((item_nr, reason))
    elif has_match(keywords):
        covered.append(item_nr)
    else:
        uncovered.append((item_nr, keywords))

print("Total items: 172")
print(f"Covered by test classes: {len(covered)}")
print(f"Personal/strategic (no code test possible): {len(personal)}")
print(f"UNCOVERED (code-testable, no matching class): {len(uncovered)}")
print()
if uncovered:
    print("UNCOVERED items:")
    for nr, kws in uncovered:
        print(f"  Item {nr}: try keywords {kws}")
else:
    print("ALL code-testable items are covered!")
print()
print("Personal/strategic items (expected no test):")
for nr, reason in personal:
    print(f"  Item {nr}: {reason}")
