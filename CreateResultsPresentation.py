import aspose.slides as slides

def create_presentation():
    models_names = ["scale_0.1_N_32_H_in_1_edo_init_Inefficient_forward_pass_embeddings_type_linear_size_10_linear_recurrent_rnn_same_seeds"]
    cols_titles = ["Arch", "Sample count", "Best Test Acc", "Train Loss (0.3,0.35)", "(0.35,0.4)", "(0.4,0.45)", "(0.45,0.5)", "(0.5,0.55)", "(0.55,0.6)", "(0.6,0.65)"]
    Architectures = ["Vanilla RNN", "Linear RNN", "Linear RNN complex diag_real_im", "Linear RNN complex stable_ring_init", "Linear RNN complex stable_ring_init_gamma"]
    sample_count = ["16","8","4","2"]
    with slides.Presentation() as pres:
        # Access first slide
        sld = pres.slides[0]

        # Define columns with widths and rows with heights
        dblCols = [60, 30, 30, 30, 30, 30, 30, 30, 30, 30]
        dblRows = [30, 60, 60, 60, 60, 60]

        # Add table shape to slide
        tbl = sld.shapes.add_table(100, 50, dblCols, dblRows)

        # Set border format for each cell
        for row in range(len(tbl.rows)):
            for cell in range(len(tbl.rows[row])):
                if row == 0:
                    tbl.rows[row][cell].text_frame.text = cols_titles[cell]
                elif cell == 0 and row > 0:
                    tbl.rows[row][cell].text_frame.text = Architectures[row-1]
                elif cell == 1 and row in range(1,5):
                    tbl.rows[row][cell].text_frame.text = sample_count[row-1]
                else:
                    tbl.rows[row][cell].text_frame.text = "Cell_" + str(cell)

                # Set border
                tbl.rows[row][cell].cell_format.border_top.fill_format.fill_type = slides.FillType.SOLID
                #tbl.rows[row][cell].cell_format.border_top.fill_format.solid_fill_color.color = drawing.Color.red
                tbl.rows[row][cell].cell_format.border_top.width = 5

                tbl.rows[row][cell].cell_format.border_bottom.fill_format.fill_type = slides.FillType.SOLID
                #tbl.rows[row][cell].cell_format.border_bottom.fill_format.solid_fill_color.color = drawing.Color.red
                tbl.rows[row][cell].cell_format.border_bottom.width = 5

                tbl.rows[row][cell].cell_format.border_left.fill_format.fill_type = slides.FillType.SOLID
                #tbl.rows[row][cell].cell_format.border_left.fill_format.solid_fill_color.color = drawing.Color.red
                tbl.rows[row][cell].cell_format.border_left.width = 5

                tbl.rows[row][cell].cell_format.border_right.fill_format.fill_type = slides.FillType.SOLID
                #tbl.rows[row][cell].cell_format.border_right.fill_format.solid_fill_color.color = drawing.Color.red
                tbl.rows[row][cell].cell_format.border_right.width = 5

        # Merge cells 1 & 2 of row 1
        #tbl.merge_cells(tbl.rows[0][0], tbl.rows[1][1], False)

        # Add text to the merged cell
        #tbl.rows[0][0].text_frame.text = "Merged Cells"

        # Save PPTX to Disk
        pres.save("table.pptx", slides.export.SaveFormat.PPTX)


create_presentation()
print("DONE")