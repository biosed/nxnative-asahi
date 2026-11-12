namespace Ryujinx.Graphics.Shader.StructuredIr
{
    readonly struct BufferDefinition
    {
        public BufferLayout Layout { get; }
        public int Set { get; }
        public int Binding { get; }
        public string Name { get; }
        public StructureType Type { get; }
        public bool IsReadOnly { get; }
        public bool IsRestrict { get; }

        public BufferDefinition(
            BufferLayout layout,
            int set,
            int binding,
            string name,
            StructureType type,
            bool isReadOnly = false,
            bool isRestrict = false)
        {
            Layout = layout;
            Set = set;
            Binding = binding;
            Name = name;
            Type = type;
            IsReadOnly = isReadOnly;
            IsRestrict = isRestrict;
        }
    }
}
