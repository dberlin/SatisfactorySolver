// Extracts every resource node's item, purity and position from an installed
// copy of Satisfactory into satisfactorysolver/data/resource_nodes.json.
//
//   dotnet run -c Release -- [game directory] [output file]
//
// The game's cooked map is read with CUE4Parse. Nodes, resource-well cores and
// satellites all live in the always-loaded Persistent_Level package.

using System.Text;
using CUE4Parse.Compression;
using CUE4Parse.FileProvider;
using CUE4Parse.MappingsProvider.Usmap;
using CUE4Parse.UE4.Assets.Exports;
using CUE4Parse.UE4.Objects.Core.Math;
using CUE4Parse.UE4.Objects.UObject;
using CUE4Parse.UE4.Versions;
using Newtonsoft.Json;

var game = args.Length > 0 ? args[0] : @"C:\Program Files (x86)\Steam\steamapps\common\Satisfactory";
var output = args.Length > 1 ? args[1] : Path.Combine("..", "..", "satisfactorysolver", "data", "resource_nodes.json");

// Item descriptor classes to the item names the solver uses.
var items = new Dictionary<string, string>
{
    ["Desc_OreIron_C"] = "Iron Ore",
    ["Desc_OreCopper_C"] = "Copper Ore",
    ["Desc_Stone_C"] = "Limestone",
    ["Desc_Coal_C"] = "Coal",
    ["Desc_OreGold_C"] = "Caterium Ore",
    ["Desc_LiquidOil_C"] = "Crude Oil",
    ["Desc_RawQuartz_C"] = "Raw Quartz",
    ["Desc_Sulfur_C"] = "Sulfur",
    ["Desc_OreBauxite_C"] = "Bauxite",
    ["Desc_OreUranium_C"] = "Uranium",
    ["Desc_NitrogenGas_C"] = "Nitrogen Gas",
    ["Desc_SAM_C"] = "SAM",
    ["Desc_Water_C"] = "Water",
};
// Actor class to whether it is a resource-well satellite. Well cores only mark
// where the pressurizer goes, and geysers feed only geothermal generators.
var nodeClasses = new Dictionary<string, bool>
{
    ["BP_ResourceNode_C"] = false,
    ["BP_FrackingSatellite_C"] = true,
};
// Unset purities take the class default, which is normal.
var purities = new Dictionary<string, string>
{
    ["RP_Inpure"] = "impure",
    ["RP_Normal"] = "normal",
    ["RP_Pure"] = "pure",
};

string? oodle = Path.Combine(AppContext.BaseDirectory, OodleHelper.OodleFileName);
if (!File.Exists(oodle)) OodleHelper.DownloadOodleDll(ref oodle);
OodleHelper.Initialize(oodle);

var provider = new DefaultFileProvider(
    Path.Combine(game, "FactoryGame", "Content", "Paks"),
    SearchOption.TopDirectoryOnly,
    new VersionContainer(EGame.GAME_UE5_6),
    StringComparer.OrdinalIgnoreCase);
provider.MappingsContainer = LoadMappings(Path.Combine(game, "CommunityResources", "FactoryGame.usmap"));
provider.Initialize();
provider.Mount();

var level = provider.LoadPackage("FactoryGame/Content/FactoryGame/Map/GameLevel01/Persistent_Level.umap");
var nodes = new List<Dictionary<string, object>>();
foreach (var actor in level.GetExports())
{
    if (!nodeClasses.TryGetValue(actor.Class?.Name.Text ?? "", out var well)) continue;
    var descriptor = actor.GetOrDefault<FPackageIndex>("mResourceClass")?.Name ?? "";
    if (!items.TryGetValue(descriptor, out var item))
        throw new InvalidDataException($"{actor.Name}: unknown resource {descriptor}");
    var purity = actor.GetOrDefault<FName>("mPurity").Text;
    var root = actor.GetOrDefault<UObject>("RootComponent")
        ?? throw new InvalidDataException($"{actor.Name}: no root component");
    var location = root.GetOrDefault<FVector>("RelativeLocation");
    nodes.Add(new()
    {
        ["resource"] = item,
        ["purity"] = purities.GetValueOrDefault(purity, "normal"),
        ["well"] = well,
        // Unreal units are centimeters.
        ["x"] = Math.Round(location.X / 100, 2),
        ["y"] = Math.Round(location.Y / 100, 2),
        ["z"] = Math.Round(location.Z / 100, 2),
    });
}
nodes = nodes
    .OrderBy(n => (string) n["resource"], StringComparer.Ordinal)
    .ThenBy(n => (bool) n["well"])
    .ThenBy(n => (double) n["x"])
    .ThenBy(n => (double) n["y"])
    .ToList();
File.WriteAllText(output, JsonConvert.SerializeObject(nodes, Formatting.Indented) + "\n");
Console.WriteLine($"Wrote {nodes.Count} resource nodes to {Path.GetFullPath(output)}");

// The game's usmap writes OptionalProperty without an inner type, which
// CUE4Parse expects, so if it does not load as is, load a copy with a
// placeholder inner type added.
static UsmapTypeMappingsProvider LoadMappings(string path)
{
    try
    {
        return new FileUsmapTypeMappingsProvider(path);
    }
    catch (Exception error) when (error is ArgumentOutOfRangeException or IndexOutOfRangeException)
    {
        var patched = Path.Combine(Path.GetTempPath(), "FactoryGame.patched.usmap");
        File.WriteAllBytes(patched, AddOptionalInnerTypes(File.ReadAllBytes(path)));
        return new FileUsmapTypeMappingsProvider(patched);
    }
}

static byte[] AddOptionalInnerTypes(byte[] data)
{
    const byte ArrayProperty = 8, StructProperty = 9, StrProperty = 10, MapProperty = 24,
        SetProperty = 25, EnumProperty = 26, OptionalProperty = 28;
    var header = new BinaryReader(new MemoryStream(data));
    if (header.ReadUInt16() != 0x30C4) throw new InvalidDataException("not a usmap");
    var version = header.ReadByte();
    if (version >= 5) throw new InvalidDataException($"unsupported usmap version {version}");
    if (version >= 1 && header.ReadInt32() != 0)
    {
        header.ReadInt64(); // package file versions
        var customVersions = header.ReadInt32();
        header.BaseStream.Position += customVersions * 20L; // GUID and version each
        header.ReadUInt32(); // net CL
    }
    if (header.ReadByte() != 0) throw new InvalidDataException("compressed usmap");
    var sizes = (int) header.BaseStream.Position;
    var size = header.ReadInt32();
    header.ReadInt32();
    var start = (int) header.BaseStream.Position;

    var r = new BinaryReader(new MemoryStream(data, start, size));
    var inserts = new List<int>();
    var nameCount = r.ReadUInt32();
    for (var i = 0; i < nameCount; i++)
    {
        var length = version >= 2 ? r.ReadUInt16() : r.ReadByte();
        r.BaseStream.Position += length;
    }
    var enumCount = r.ReadUInt32();
    for (var i = 0; i < enumCount; i++)
    {
        r.ReadInt32();
        long values = version >= 3 ? r.ReadUInt16() : r.ReadByte();
        r.BaseStream.Position += values * (version >= 4 ? 12L : 4L);
    }
    void PropertyType()
    {
        switch (r.ReadByte())
        {
            case EnumProperty: PropertyType(); r.ReadInt32(); break;
            case StructProperty: r.ReadInt32(); break;
            case ArrayProperty or SetProperty: PropertyType(); break;
            case MapProperty: PropertyType(); PropertyType(); break;
            case OptionalProperty: inserts.Add((int) r.BaseStream.Position); break;
        }
    }
    var structCount = r.ReadUInt32();
    for (var i = 0; i < structCount; i++)
    {
        r.ReadInt64(); // name, super name
        r.ReadUInt16();
        var properties = r.ReadUInt16();
        for (var j = 0; j < properties; j++)
        {
            r.BaseStream.Position += 7; // index, array size, name
            PropertyType();
        }
    }

    var body = new MemoryStream();
    var last = 0;
    foreach (var at in inserts)
    {
        body.Write(data, start + last, at - last);
        body.WriteByte(StrProperty);
        last = at;
    }
    body.Write(data, start + last, size - last);
    var result = new MemoryStream();
    result.Write(data, 0, sizes);
    var w = new BinaryWriter(result, Encoding.UTF8, leaveOpen: true);
    w.Write((int) body.Length);
    w.Write((int) body.Length);
    body.WriteTo(result);
    result.Write(data, start + size, data.Length - start - size);
    return result.ToArray();
}
